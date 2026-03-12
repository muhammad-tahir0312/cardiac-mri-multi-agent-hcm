"""
src/data/cardiac_dataset.py

PyTorch Dataset for the Cardiac MRI HCM Multi-Agent System.

Two complementary dataset classes are provided:

``CardiacMRIDataset``
    Loads **pre-processed NIfTI volumes** from ``data/processed/``.
    Each row in the split CSV maps directly to one ``.nii.gz`` file.
    Preferred for training — volumes are loaded and cached on first access.

``CardiacMRIRawDataset``
    End-to-end dataset that combines the :class:`RouterAgent` and
    :class:`IngestionAgent` and :class:`PreprocessingAgent` in one
    ``__getitem__`` call.  Convenient for rapid prototyping or when
    disk space for pre-processed volumes is scarce.

Both datasets return ``(tensor, label)`` pairs where:
- ``tensor`` — float32, shape ``(1, D, H, W)``
- ``label``  — int64 scalar (0 = Normal, 1 = Sick)

Usage
-----
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    from src.data.cardiac_dataset import CardiacMRIDataset, build_dataloaders

    cfg = OmegaConf.load("src/config/base.yaml")
    loaders = build_dataloaders(cfg)
    for volume, label in loaders["train"]:
        ...   # volume: (B, 1, D, H, W)  label: (B,)
"""

import logging
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

logger = logging.getLogger(__name__)

# Lazy import — only needed by CardiacMRIRawDataset
_agents_imported = False


def _import_agents():
    global _agents_imported
    if not _agents_imported:
        global RouterAgent, IngestionAgent, PreprocessingAgent
        from src.agents.router_agent import RouterAgent  # noqa: F401
        from src.agents.ingestion_agent import IngestionAgent  # noqa: F401
        from src.agents.preprocessing_agent import PreprocessingAgent  # noqa: F401
        _agents_imported = True


# ---------------------------------------------------------------------------
# CardiacMRIDataset — pre-processed NIfTI volumes
# ---------------------------------------------------------------------------


class CardiacMRIDataset(Dataset):
    """Dataset over pre-processed ``.nii.gz`` volumes.

    Expects that ``scripts/04_preprocess_dataset.py`` (or equivalent) has
    already converted each DICOM series into a fixed-size NIfTI volume stored
    under ``processed_root``.

    The split CSV (``train.csv`` / ``val.csv`` / ``test.csv``) must have at
    minimum the columns: ``patient_id``, ``class``, ``series_path``.
    An optional ``processed_path`` column is used if present; otherwise the
    path is inferred from ``processed_root``.

    Args:
        split_csv:       Path to the split CSV file.
        processed_root:  Root directory containing ``.nii.gz`` files.
        class_to_idx:    Mapping from class name to integer label.
        transform:       Optional callable applied to the loaded tensor
                         *after* loading, e.g. additional augmentation.
        augment:         If ``True``, pass ``augment=True`` to the
                         :class:`~src.agents.preprocessing_agent.PreprocessingAgent`
                         (only relevant when ``transform`` is a
                         ``PreprocessingAgent`` instance).
        cache_in_memory: Pre-load all volumes into RAM on construction.
                         Speeds up epoch iteration; requires sufficient RAM.

    Raises:
        FileNotFoundError: If ``split_csv`` does not exist.
        ValueError:        If required columns are missing from the CSV.
    """

    REQUIRED_COLUMNS = {"patient_id", "class", "series_path"}

    def __init__(
        self,
        split_csv: Path,
        processed_root: Path,
        class_to_idx: Dict[str, int],
        transform: Optional[Callable] = None,
        augment: bool = False,
        cache_in_memory: bool = False,
    ) -> None:
        if not split_csv.exists():
            raise FileNotFoundError(f"Split CSV not found: {split_csv}")

        self.df = pd.read_csv(split_csv)
        missing = self.REQUIRED_COLUMNS - set(self.df.columns)
        if missing:
            raise ValueError(f"Split CSV missing required columns: {missing}")

        self.processed_root = processed_root
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.augment = augment

        # Deduplicate to one row per patient (series-level CSV → patient-level)
        self.df = self.df.drop_duplicates(subset=["patient_id"]).reset_index(drop=True)
        logger.info(
            "CardiacMRIDataset loaded: %d samples from %s", len(self.df), split_csv.name
        )

        self._cache: Optional[Dict[int, Tuple[torch.Tensor, int]]] = None
        if cache_in_memory:
            self._build_cache()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._cache is not None:
            tensor, label = self._cache[idx]
            return tensor, torch.tensor(label, dtype=torch.long)

        row = self.df.iloc[idx]
        nifti_path = self._resolve_nifti_path(row)
        tensor = self._load_nifti(nifti_path)
        label = self.class_to_idx[row["class"]]

        if self.transform is not None:
            tensor = self.transform(tensor)

        return tensor, torch.tensor(label, dtype=torch.long)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def labels(self) -> np.ndarray:
        """Array of integer labels, one per sample (for sampler construction).

        Returns:
            1-D int64 numpy array.
        """
        return np.array(
            [self.class_to_idx[c] for c in self.df["class"]], dtype=np.int64
        )

    @property
    def class_counts(self) -> Dict[str, int]:
        """Sample counts per class.

        Returns:
            Dict mapping class name to count.
        """
        return self.df["class"].value_counts().to_dict()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_nifti_path(self, row: pd.Series) -> Path:
        """Derive the NIfTI path from a CSV row.

        Checks for a ``processed_path`` column first; falls back to building
        the path from ``processed_root``, the class label, and the patient ID.

        Args:
            row: A single row from the split DataFrame.

        Returns:
            Path object pointing to the ``.nii.gz`` file.
        """
        if "processed_path" in row.index and pd.notna(row["processed_path"]):
            return Path(row["processed_path"])

        # Infer: processed_root / class / patient_dir_name.nii.gz
        patient_id: str = row["patient_id"]  # e.g. "Normal/Directory_1"
        # Replace path separator so it becomes a valid filename component
        safe_name = patient_id.replace("/", "_").replace("\\", "_")
        return self.processed_root / f"{safe_name}.nii.gz"

    @staticmethod
    def _load_nifti(path: Path) -> torch.Tensor:
        """Load a NIfTI volume as a float32 tensor with shape ``(1, D, H, W)``.

        Args:
            path: Path to a ``.nii.gz`` file.

        Returns:
            Float32 tensor with shape ``(1, D, H, W)``.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if not path.exists():
            raise FileNotFoundError(f"Processed NIfTI not found: {path}")
        nifti = nib.load(str(path))
        data: np.ndarray = nifti.get_fdata(dtype=np.float32)

        if data.ndim != 3:
            raise ValueError(
                f"Expected 3-D NIfTI volume for {path}, got shape {data.shape}"
            )

        # Keep the depth axis first. Most generated files in this project are
        # already saved as (D, H, W), but this branch also supports (H, W, D).
        d0, d1, d2 = data.shape
        if d0 <= d1 and d0 <= d2:
            data_dhw = data
        elif d2 <= d0 and d2 <= d1:
            data_dhw = np.transpose(data, (2, 0, 1))
        else:
            logger.warning(
                "Ambiguous NIfTI axis order for %s with shape %s; assuming (D,H,W).",
                path,
                data.shape,
            )
            data_dhw = data

        tensor = torch.from_numpy(np.ascontiguousarray(data_dhw)).unsqueeze(0)    # → (1, D, H, W)
        return tensor

    def _build_cache(self) -> None:
        """Pre-load all volumes into CPU RAM.

        Sets ``self._cache`` to a dict mapping sample index to
        ``(tensor, label)`` tuple.
        """
        logger.info("Building in-memory cache for %d samples …", len(self.df))
        self._cache = {}
        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            nifti_path = self._resolve_nifti_path(row)
            tensor = self._load_nifti(nifti_path)
            label = self.class_to_idx[row["class"]]
            if self.transform is not None:
                tensor = self.transform(tensor)
            self._cache[idx] = (tensor, label)
        logger.info("Cache built.")


# ---------------------------------------------------------------------------
# CardiacMRIRawDataset — end-to-end from DICOM on-the-fly
# ---------------------------------------------------------------------------


class CardiacMRIRawDataset(Dataset):
    """End-to-end dataset: DICOM → NIfTI → preprocessed tensor.

    Combines :class:`~src.agents.router_agent.RouterAgent`,
    :class:`~src.agents.ingestion_agent.IngestionAgent`, and
    :class:`~src.agents.preprocessing_agent.PreprocessingAgent` inside
    ``__getitem__``.  No intermediate files are written to disk.

    This is slower per epoch than :class:`CardiacMRIDataset` (reads raw DICOMs
    every time) but requires no pre-processing step.

    Args:
        split_csv:    Path to the split CSV with columns
                      ``patient_id | class | series_path | image_count``.
        cfg:          OmegaConf config object.
        augment:      If ``True``, apply TorchIO augmentation.

    Raises:
        FileNotFoundError: If ``split_csv`` does not exist.
    """

    def __init__(
        self,
        split_csv: Path,
        cfg: DictConfig,
        augment: bool = False,
    ) -> None:
        _import_agents()

        if not split_csv.exists():
            raise FileNotFoundError(f"Split CSV not found: {split_csv}")

        self.df = pd.read_csv(split_csv)
        # One row per unique patient directory
        self.df = self.df.drop_duplicates(subset=["patient_id"]).reset_index(drop=True)
        self.cfg = cfg
        self.augment = augment
        self.class_to_idx: Dict[str, int] = dict(cfg.data.class_to_idx)

        self._router = RouterAgent(
            min_slices=int(cfg.router.min_slices),
            priority_keywords=list(cfg.router.priority_keywords),
        )
        self._ingestor = IngestionAgent(rescale=bool(cfg.preprocessing.rescale_dicom))
        self._preprocessor = PreprocessingAgent(cfg)

        logger.info(
            "CardiacMRIRawDataset: %d patients, augment=%s", len(self.df), augment
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        patient_dir = Path(row["series_path"]).parent  # series_path → patient dir

        # 1. Route: pick best series
        series_path = self._router.select_series(patient_dir)
        if series_path is None:
            # Fallback: use the series_path column directly
            series_path = Path(row["series_path"])

        # 2. Ingest: DICOM → (H, W, D) numpy
        volume, _affine, _meta = self._ingestor.load_series(series_path)

        # 3. Preprocess → (1, D, H, W) tensor
        tensor = self._preprocessor.preprocess(volume, augment=self.augment)

        label = self.class_to_idx[row["class"]]
        return tensor, torch.tensor(label, dtype=torch.long)


# ---------------------------------------------------------------------------
# Sampler helpers
# ---------------------------------------------------------------------------


def build_weighted_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    """Create a :class:`~torch.utils.data.WeightedRandomSampler` for imbalanced data.

    Each sample's weight is the inverse of its class frequency, so all classes
    contribute equally per training epoch.

    Args:
        labels: 1-D integer array of sample labels.

    Returns:
        :class:`~torch.utils.data.WeightedRandomSampler` instance.
    """
    class_counts = np.bincount(labels)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = torch.tensor(class_weights[labels], dtype=torch.double)
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """Compute per-class weights for use in a weighted loss function.

    Weights are inversely proportional to class frequency and normalised so
    they sum to ``num_classes``.

    Args:
        labels:      1-D integer label array from a training split.
        num_classes: Total number of classes.

    Returns:
        Float32 tensor of shape ``(num_classes,)``.
    """
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    weights = 1.0 / np.maximum(counts, 1)
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Factory: build_dataloaders
# ---------------------------------------------------------------------------


def build_dataloaders(
    cfg: DictConfig,
    use_raw: bool = False,
    cache_train: bool = False,
) -> Dict[str, DataLoader]:
    """Build train / val / test :class:`~torch.utils.data.DataLoader` objects.

    Args:
        cfg:         OmegaConf config (``src/config/base.yaml``).
        use_raw:     If ``True``, use :class:`CardiacMRIRawDataset` (DICOM
                     on-the-fly). Otherwise use :class:`CardiacMRIDataset`
                     (pre-processed NIfTI).
        cache_train: Pre-load training volumes into RAM (ignored when
                     ``use_raw=True``).

    Returns:
        Dict with keys ``"train"``, ``"val"``, ``"test"`` mapping to
        :class:`~torch.utils.data.DataLoader` instances.
    """
    splits_dir    = Path(cfg.paths.splits)
    processed_dir = Path(cfg.paths.data_processed)
    class_to_idx: Dict[str, int] = dict(cfg.data.class_to_idx)
    imbalance_strategy = str(cfg.training.get("imbalance_strategy", "loss_weights")).lower()

    if imbalance_strategy not in {"loss_weights", "sampler", "none"}:
        raise ValueError(
            f"Unknown training.imbalance_strategy '{imbalance_strategy}'. "
            "Choose from: loss_weights, sampler, none."
        )

    from src.agents.preprocessing_agent import PreprocessingAgent
    preprocessor = PreprocessingAgent(cfg)

    loaders: Dict[str, DataLoader] = {}

    for split in ("train", "val", "test"):
        csv_path = splits_dir / f"{split}.csv"
        is_train = split == "train"

        if use_raw:
            dataset: Dataset = CardiacMRIRawDataset(
                split_csv=csv_path,
                cfg=cfg,
                augment=is_train,
            )
        else:
            def _train_tensor_transform(tensor: torch.Tensor) -> torch.Tensor:
                if preprocessor.train_transforms is None:
                    return tensor
                return preprocessor._apply_torchio(tensor)

            dataset = CardiacMRIDataset(
                split_csv=csv_path,
                processed_root=processed_dir,
                class_to_idx=class_to_idx,
                transform=(_train_tensor_transform if is_train else None),
                augment=is_train,
                cache_in_memory=(cache_train and is_train),
            )

        if is_train and imbalance_strategy == "sampler":
            sampler = build_weighted_sampler(dataset.labels)
            shuffle = False
        else:
            sampler = None
            shuffle = is_train  # val/test must stay ordered

        loader = DataLoader(
            dataset,
            batch_size=int(cfg.training.batch_size),
            shuffle=(shuffle and sampler is None),
            sampler=sampler,
            num_workers=int(cfg.dataset.num_workers),
            pin_memory=bool(cfg.dataset.pin_memory) and torch.cuda.is_available(),
            prefetch_factor=int(cfg.dataset.prefetch_factor) if cfg.dataset.num_workers > 0 else None,
            drop_last=is_train,
        )
        loaders[split] = loader
        logger.info(
            "DataLoader '%s': %d samples, batch_size=%d, weighted_sampler=%s, imbalance_strategy=%s",
            split, len(dataset), cfg.training.batch_size, sampler is not None, imbalance_strategy,
        )

    return loaders
