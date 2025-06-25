# =========================================================================
# RECOMMENDED PROJECT STRUCTURE FOR MASTER'S RESEARCH
# =========================================================================

"""
my_masters_project/
├── README.md                          # Project overview and setup
├── requirements.txt                   # Dependencies
├── environment.yml                    # Conda environment
├── .gitignore                        # Git ignore rules
├── config/                           # Configuration files
│   ├── __init__.py
│   ├── settings.py                   # Global settings
│   ├── paths.py                      # File paths configuration
│   └── parameters.py                 # Analysis parameters
├── data/                             # Data directory (gitignored)
│   ├── raw/                          # Original, immutable data
│   ├── interim/                      # Intermediate processed data
│   ├── processed/                    # Final datasets
│   └── external/                     # External reference data
├── src/                              # Source code
│   ├── __init__.py
│   ├── data/                         # Data handling modules
│   │   ├── __init__.py
│   │   ├── loaders.py               # Data loading functions
│   │   ├── validators.py            # Data quality checks
│   │   └── exporters.py             # Data export functions
│   ├── processing/                   # Processing modules by phase
│   │   ├── __init__.py
│   │   ├── phase01_quality.py       # Data quality evaluation
│   │   ├── phase02_correction.py    # Temperature correction
│   │   ├── phase03_inversion.py     # Inversion and plotting
│   │   └── phase04_clustering.py    # TL-ERT clustering
│   ├── models/                       # Model implementations
│   │   ├── __init__.py
│   │   ├── chambers.py              # Chambers model
│   │   ├── base_model.py            # Base model class
│   │   └── model_registry.py        # Model version registry
│   ├── visualization/                # Plotting and visualization
│   │   ├── __init__.py
│   │   ├── plotters.py              # General plotting functions
│   │   └── reports.py               # Report generation
│   └── utils/                        # Utility functions
│       ├── __init__.py
│       ├── io.py                    # I/O helpers
│       ├── math_utils.py            # Mathematical utilities
│       └── version_control.py       # Version management
├── notebooks/                        # Jupyter notebooks
│   ├── 01_exploratory/              # Initial data exploration
│   ├── 02_development/              # Method development
│   ├── 03_analysis/                 # Main analysis notebooks
│   └── 04_results/                  # Final results and figures
├── scripts/                          # Executable scripts
│   ├── run_phase01.py               # Phase 1 runner
│   ├── run_phase02.py               # Phase 2 runner
│   ├── run_phase03.py               # Phase 3 runner
│   ├── run_phase04.py               # Phase 4 runner
│   ├── run_full_pipeline.py         # Complete pipeline
│   └── utilities/                   # Utility scripts
├── tests/                            # Unit tests
│   ├── __init__.py
│   ├── test_data/                   # Test data
│   ├── test_processing/             # Processing tests
│   └── test_models/                 # Model tests
├── results/                          # Analysis results (gitignored)
│   ├── phase01/                     # Quality evaluation results
│   ├── phase02/                     # Correction results
│   ├── phase03/                     # Inversion results
│   ├── phase04/                     # Clustering results
│   └── final/                       # Final results for thesis
└── docs/                            # Documentation
    ├── methodology.md               # Methods documentation
    ├── api_reference.md             # Code documentation
    └── changelog.md                 # Version history
"""

# =========================================================================
# KEY ARCHITECTURE PRINCIPLES
# =========================================================================

# 1. VERSION CONTROL & BACKWARD COMPATIBILITY
# =========================================================================

# config/settings.py
import os
from pathlib import Path

class ProjectConfig:
    """Central configuration management with versioning"""
    
    # Version control
    PROJECT_VERSION = "1.2.0"
    DATA_VERSION = "v2"
    MODEL_VERSION = "v1.1"
    
    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    RESULTS_DIR = PROJECT_ROOT / "results"
    
    # Backward compatibility settings
    LEGACY_SUPPORT = True
    MIN_SUPPORTED_VERSION = "1.0.0"
    
    # Analysis parameters with versioning
    ANALYSIS_PARAMS = {
        "v1.0": {
            "chambers_model": {"hour_step": 24, "bounds": [0.1, 10]},
            "quality_thresholds": {"r2_min": 0.7, "rmse_max": 2.0}
        },
        "v1.1": {
            "chambers_model": {"hour_step": 24, "bounds": [0.1, 15]},
            "quality_thresholds": {"r2_min": 0.8, "rmse_max": 1.5}
        }
    }
    
    @classmethod
    def get_params(cls, version=None):
        """Get parameters for specific version"""
        version = version or cls.MODEL_VERSION
        return cls.ANALYSIS_PARAMS.get(version, cls.ANALYSIS_PARAMS["v1.1"])

# =========================================================================
# 2. MODULAR PROCESSING WITH INHERITANCE
# =========================================================================

# src/processing/base_processor.py
from abc import ABC, abstractmethod
import logging
from datetime import datetime
import pickle
import json

class BaseProcessor(ABC):
    """Base class for all processing phases"""
    
    def __init__(self, config, version=None):
        self.config = config
        self.version = version or config.MODEL_VERSION
        self.logger = self._setup_logging()
        self.results = {}
        
    def _setup_logging(self):
        """Setup logging for this processor"""
        logger = logging.getLogger(f"{self.__class__.__name__}")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    @abstractmethod
    def process(self, data, **kwargs):
        """Main processing method - must be implemented by subclasses"""
        pass
    
    def save_results(self, output_path, format='pickle'):
        """Save processing results with version info"""
        results_with_meta = {
            'results': self.results,
            'metadata': {
                'processor': self.__class__.__name__,
                'version': self.version,
                'timestamp': datetime.now().isoformat(),
                'config_version': self.config.PROJECT_VERSION
            }
        }
        
        if format == 'pickle':
            with open(f"{output_path}.pkl", 'wb') as f:
                pickle.dump(results_with_meta, f)
        elif format == 'json':
            with open(f"{output_path}.json", 'w') as f:
                json.dump(results_with_meta, f, indent=2, default=str)
        
        self.logger.info(f"Results saved to {output_path}")
    
    def load_results(self, input_path):
        """Load results with version checking"""
        if input_path.endswith('.pkl'):
            with open(input_path, 'rb') as f:
                data = pickle.load(f)
        else:
            with open(input_path, 'r') as f:
                data = json.load(f)
        
        # Version compatibility check
        if 'metadata' in data:
            file_version = data['metadata'].get('version', '1.0.0')
            if not self._is_compatible_version(file_version):
                self.logger.warning(f"Version mismatch: file {file_version}, current {self.version}")
        
        return data.get('results', data)
    
    def _is_compatible_version(self, file_version):
        """Check if file version is compatible"""
        # Simple version compatibility logic
        file_major = int(file_version.split('.')[0])
        current_major = int(self.version.split('.')[0])
        return file_major == current_major

# =========================================================================
# 3. PHASE-SPECIFIC PROCESSORS
# =========================================================================

# src/processing/phase01_quality.py
from .base_processor import BaseProcessor
import pandas as pd
import numpy as np

class DataQualityProcessor(BaseProcessor):
    """Phase 1: Data Quality Evaluation"""
    
    def process(self, data, **kwargs):
        """Evaluate data quality"""
        self.logger.info("Starting data quality evaluation")
        
        params = self.config.get_params(self.version)
        quality_params = params['quality_thresholds']
        
        # Data quality checks
        quality_results = {}
        
        for column in data.columns:
            if 'Temp' in column:
                col_data = data[column].dropna()
                quality_results[column] = {
                    'missing_percentage': data[column].isna().mean() * 100,
                    'outlier_count': self._count_outliers(col_data),
                    'range': (col_data.min(), col_data.max()),
                    'std': col_data.std(),
                    'quality_score': self._calculate_quality_score(col_data)
                }
        
        self.results = {
            'quality_metrics': quality_results,
            'parameters_used': quality_params,
            'overall_quality': self._overall_quality_assessment(quality_results)
        }
        
        self.logger.info(f"Quality evaluation complete. Overall score: {self.results['overall_quality']:.2f}")
        return self.results
    
    def _count_outliers(self, data, method='iqr'):
        """Count outliers using IQR method"""
        if method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return ((data < lower_bound) | (data > upper_bound)).sum()
        return 0
    
    def _calculate_quality_score(self, data):
        """Calculate quality score (0-1)"""
        missing_penalty = min(data.isna().mean() * 2, 1)  # Cap at 1
        outlier_penalty = min(self._count_outliers(data) / len(data) * 5, 1)
        return max(0, 1 - missing_penalty - outlier_penalty)
    
    def _overall_quality_assessment(self, quality_results):
        """Calculate overall quality score"""
        scores = [result['quality_score'] for result in quality_results.values()]
        return np.mean(scores) if scores else 0

# src/processing/phase02_correction.py
from .base_processor import BaseProcessor

class TemperatureCorrectionProcessor(BaseProcessor):
    """Phase 2: Temperature Correction"""
    
    def process(self, data, quality_results=None, **kwargs):
        """Apply temperature corrections"""
        self.logger.info("Starting temperature correction")
        
        # Load quality results if provided
        if quality_results is None and hasattr(self, 'load_previous_phase'):
            quality_results = self.load_previous_phase('phase01')
        
        corrected_data = data.copy()
        
        # Apply corrections based on quality assessment
        correction_log = {}
        
        for column in data.columns:
            if 'Temp' in column and quality_results:
                quality_info = quality_results.get('quality_metrics', {}).get(column)
                if quality_info and quality_info['quality_score'] < 0.8:
                    # Apply correction methods
                    corrected_data[column] = self._apply_corrections(
                        data[column], quality_info
                    )
                    correction_log[column] = "Applied outlier removal and smoothing"
        
        self.results = {
            'corrected_data': corrected_data,
            'correction_log': correction_log,
            'correction_summary': self._correction_summary(correction_log)
        }
        
        return self.results
    
    def _apply_corrections(self, series, quality_info):
        """Apply specific corrections to temperature series"""
        # Remove outliers
        corrected = self._remove_outliers(series)
        
        # Apply smoothing if needed
        if quality_info['std'] > 5.0:  # High variability
            corrected = corrected.rolling(window=3, center=True).mean().fillna(corrected)
        
        return corrected
    
    def _remove_outliers(self, series):
        """Remove outliers using IQR method"""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Replace outliers with NaN then interpolate
        cleaned = series.copy()
        cleaned[(cleaned < lower_bound) | (cleaned > upper_bound)] = np.nan
        return cleaned.interpolate()
    
    def _correction_summary(self, correction_log):
        """Generate correction summary"""
        return {
            'columns_corrected': len(correction_log),
            'total_columns': len([col for col in correction_log.keys() if 'Temp' in col])
        }

# =========================================================================
# 4. PIPELINE ORCHESTRATION
# =========================================================================

# scripts/run_full_pipeline.py
from src.processing.phase01_quality import DataQualityProcessor
from src.processing.phase02_correction import TemperatureCorrectionProcessor
from src.processing.phase03_inversion import InversionProcessor
from src.processing.phase04_clustering import ClusteringProcessor
from src.data.loaders import load_TDR_data
from config.settings import ProjectConfig
import argparse

class MastersPipeline:
    """Complete analysis pipeline with version control"""
    
    def __init__(self, config_version=None, force_rerun=False):
        self.config = ProjectConfig()
        self.config_version = config_version or self.config.PROJECT_VERSION
        self.force_rerun = force_rerun
        
        # Initialize processors
        self.processors = {
            'phase01': DataQualityProcessor(self.config),
            'phase02': TemperatureCorrectionProcessor(self.config),
            'phase03': InversionProcessor(self.config),
            'phase04': ClusteringProcessor(self.config)
        }
    
    def run_pipeline(self, data_path, phases=None, save_intermediate=True):
        """Run the complete or partial pipeline"""
        phases = phases or ['phase01', 'phase02', 'phase03', 'phase04']
        
        # Load data
        data = load_TDR_data(data_path)
        results = {}
        
        for phase in phases:
            print(f"\n{'='*50}")
            print(f"Running {phase.upper()}")
            print(f"{'='*50}")
            
            # Check if results already exist
            phase_results_path = self.config.RESULTS_DIR / phase / "results.pkl"
            
            if not self.force_rerun and phase_results_path.exists():
                print(f"Loading existing results for {phase}")
                results[phase] = self.processors[phase].load_results(phase_results_path)
            else:
                # Run processing
                if phase == 'phase01':
                    results[phase] = self.processors[phase].process(data)
                elif phase == 'phase02':
                    results[phase] = self.processors[phase].process(
                        data, quality_results=results.get('phase01')
                    )
                elif phase == 'phase03':
                    corrected_data = results['phase02']['corrected_data']
                    results[phase] = self.processors[phase].process(corrected_data)
                elif phase == 'phase04':
                    inversion_results = results['phase03']
                    results[phase] = self.processors[phase].process(inversion_results)
                
                # Save intermediate results
                if save_intermediate:
                    phase_results_path.parent.mkdir(parents=True, exist_ok=True)
                    self.processors[phase].save_results(
                        str(phase_results_path.with_suffix(''))
                    )
        
        return results
    
    def run_single_phase(self, phase, data_path, **kwargs):
        """Run a single phase of the pipeline"""
        return self.run_pipeline(data_path, phases=[phase], **kwargs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Masters Project Pipeline')
    parser.add_argument('--data_path', required=True, help='Path to data file')
    parser.add_argument('--phases', nargs='+', 
                       choices=['phase01', 'phase02', 'phase03', 'phase04'],
                       default=['phase01', 'phase02', 'phase03', 'phase04'],
                       help='Phases to run')
    parser.add_argument('--force_rerun', action='store_true',
                       help='Force rerun even if results exist')
    parser.add_argument('--config_version', 
                       help='Configuration version to use')
    
    args = parser.parse_args()
    
    # Run pipeline
    pipeline = MastersPipeline(
        config_version=args.config_version,
        force_rerun=args.force_rerun
    )
    
    results = pipeline.run_pipeline(
        data_path=args.data_path,
        phases=args.phases
    )
    
    print("\n" + "="*50)
    print("PIPELINE COMPLETE")
    print("="*50)
    for phase, result in results.items():
        print(f"{phase}: ✓")

# =========================================================================
# 5. USAGE EXAMPLES
# =========================================================================

"""
# Command line usage:
python scripts/run_full_pipeline.py --data_path data/raw/sensors.xlsx

# Run specific phases:
python scripts/run_full_pipeline.py --data_path data/raw/sensors.xlsx --phases phase01 phase02

# Force rerun with specific version:
python scripts/run_full_pipeline.py --data_path data/raw/sensors.xlsx --force_rerun --config_version v1.0

# In Python/Jupyter:
from scripts.run_full_pipeline import MastersPipeline

pipeline = MastersPipeline()
results = pipeline.run_pipeline('data/raw/sensors.xlsx')

# Run single phase:
quality_results = pipeline.run_single_phase('phase01', 'data/raw/sensors.xlsx')
"""