"""Jax-PWA public API."""

# JAX reads GPU allocator environment variables when its backend initializes.
# Set the project default before importing any submodule that imports JAX. Using
# setdefault keeps an explicit user/shell configuration authoritative.
import os as _os

_os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from . import config as _config

# The project deliberately runs float64/complex128 throughout the numerical
# pipeline for amplitude-analysis stability; this is not JAX's own default.
# Enable it on import so callers don't need a separate enable_x64() call before
# any numerical work. An explicit JAX_ENABLE_X64 set before import is left
# authoritative; enable_x64(False) remains available to opt back out.
if _os.environ.get("JAX_ENABLE_X64") is None:
    _config.enable_x64()

from .amplitude import (
    AmplitudeComponent,
    CoherentAmplitudeModel,
    ConstantAmplitude,
    PreparedAmplitudeCache,
)
from .background import (
    BackgroundCategory, CPBackgroundCategory, ChargeScaledBackground,
    charge_scaled_background,
)
from .coefficients import CPRealImag, RealImag
from .config import enable_x64
from .constraints import ConstrainedNLL, GaussianConstraint, QMISmoothnessConstraint
from .cp_workflow import CPBackgroundSpec, CPFitSession
from .decay import DalitzAmplitude, DecayChannel, DecayModel, NonResonant, Resonance
from .discriminants import (
    BreitWigner1D,
    Chebyshev1D,
    CrystalBall1D,
    Exponential1D,
    FactorizedDensity,
    Gaussian1D,
    Histogram1D,
    LineshapeIntensity1D,
    SumPDF1D,
)
from .dynamics import (
    LASS,
    QMI,
    QMI2D,
    BaBarFlatte,
    CascadeChain,
    CovariantAngular,
    Flatte,
    GooFitLegacyAngular,
    GounarisSakurai,
    Isobar,
    KMatrix,
    PairChain,
    PipiKKRescattering,
    PolarFormFactorSymNR,
    Pole,
    RelativisticBreitWigner,
    Rescattering2,
    ResonanceAmplitude,
    ResonanceContext,
    RhoOmegaMixing,
    SigmaPole,
    SympyLineshape,
    Zemach_P,
    Zemach_Pstar,
    ZemachP,
    ZemachPstar,
    physical_bin_mask,
)
from .fit import Minimizer, MultiStartResult, NesterovResult, Parameter, ParameterKind
from .four_body import FourBodyDecayModel, NBodyDecayChannel
from .goodness_of_fit import (
    BinnedChi2Result,
    PointToPointResult,
    chi2_from_histograms,
    kdtree_local_residuals,
    point_to_point_dissimilarity,
)
from .integration import DalitzGaussLegendreGrid
from .io import (
    cp_models_from_spec,
    cp_models_to_spec,
    cp_models_with_fitted_values,
    export_cp_models,
    export_model,
    histogram_background_from_root,
    histogram_efficiency_from_root,
    import_cp_models,
    import_model,
    model_from_spec,
    model_to_spec,
    model_with_fitted_values,
    read_phase_space_sample,
    read_root_histogram2d,
    read_root_tree,
    square_dalitz_background_from_root,
    square_dalitz_efficiency_from_root,
    write_cp_phase_space_sample,
    write_phase_space_sample,
    write_phase_space_samples,
)
from .kinematics import (
    CovariantKinematics,
    NBodyPhaseSpaceMC,
    NBodySample,
    PhaseSpaceMC,
    PhaseSpaceSample,
    SquareDalitzGrid,
    boost_to_rest_frame,
    cascade_coordinates,
    covariant_kinematics,
    covariant_kinematics_from_invariants,
    dalitz_s13_limits,
    fold_thetaprime,
    invariants_to_square_dalitz,
    pair_coordinates,
    pair_coordinates_to_momenta,
    square_dalitz_jacobian,
    square_dalitz_to_invariants,
)
from .likelihood import (
    CPJointNLL,
    MultiBackgroundNLL,
    NeutralMesonMixing,
    TimeDependentBackgroundCategory,
    TimeDependentDalitzNLL,
    TimeDependentMixtureNLL,
    WeightedUnbinnedNLL,
    YieldAsymmetry,
)
from .observables import delta_method_covariance, delta_method_errors, delta_method_jacobian
from .pdf import SCFSignalPDF, SignalPDF
from .plotting import (
    binned_data,
    plot_binned_data,
    plot_contour,
    plot_dalitz,
    plot_pulls,
    plot_square_dalitz,
)
from .resolution import (
    ConvolvedPDF1D,
    GaussianResolution1D,
    SparseMigration,
    SquareDalitzSCFMap,
)
from .projection_toys import CPProjectionToy
from .sampling import weighted_resample
from .square_histograms import (
    SquareDalitzHistogramBackground,
    SquareDalitzHistogramEfficiency,
)
from .time_dependent_workflow import (
    TimeDependentBackgroundSpec,
    TimeDependentFitSession,
)
from .time_dependent_toy import TimeDependentToy, generate_time_dependent_toy
from .toy_api import (
    CPToyBackground,
    PreparedInverseToyGenerator,
    ToyBackground,
    generate_cp_toy,
    generate_signal_toy,
    generate_toy,
    prepare_inverse_toy_generator,
)
from .veto import (
    CompositeVeto,
    FunctionalVeto,
    MassWindowVeto,
    VetoedDensity,
    VetoMap,
    vetoed_signal_pdf,
)
from .workflow import BackgroundSpec, FitSession

__all__ = [
    "CascadeChain", "FourBodyDecayModel", "Isobar", "NBodyDecayChannel",
    "NBodyPhaseSpaceMC", "NBodySample", "PairChain",
    "cascade_coordinates", "pair_coordinates", "pair_coordinates_to_momenta",
    "NeutralMesonMixing",
    "TimeDependentDalitzNLL",
    "TimeDependentBackgroundCategory",
    "TimeDependentMixtureNLL",
    "TimeDependentBackgroundSpec",
    "TimeDependentFitSession",
    "TimeDependentToy",
    "generate_time_dependent_toy",
    "AmplitudeComponent", "BackgroundCategory", "BackgroundSpec", "BinnedChi2Result", "BreitWigner1D", "CPBackgroundCategory", "kdtree_local_residuals",
    "CPBackgroundSpec", "CPFitSession", "CPProjectionToy", "CPJointNLL",
    "CPToyBackground", "BaBarFlatte", "CPRealImag",
    "CoherentAmplitudeModel", "CompositeVeto", "ConstrainedNLL", "ConstantAmplitude",
    "Chebyshev1D", "ConvolvedPDF1D", "CovariantAngular", "CovariantKinematics", "CrystalBall1D", "DalitzAmplitude",
    "DecayChannel", "DecayModel", "Exponential1D", "FactorizedDensity", "FitSession",
    "Flatte", "FunctionalVeto", "Gaussian1D", "GaussianConstraint", "GaussianResolution1D",
    "GooFitLegacyAngular", "GounarisSakurai", "Histogram1D", "KMatrix", "LASS", "LineshapeIntensity1D", "DalitzGaussLegendreGrid",
    "MassWindowVeto", "Minimizer", "MultiBackgroundNLL", "MultiStartResult", "NesterovResult", "NonResonant",
    "Parameter", "ParameterKind", "PhaseSpaceMC", "PhaseSpaceSample", "Pole", "PointToPointResult",
    "PreparedAmplitudeCache", "PreparedInverseToyGenerator", "PipiKKRescattering", "PolarFormFactorSymNR", "QMI", "QMI2D", "QMISmoothnessConstraint", "RealImag", "RelativisticBreitWigner", "RhoOmegaMixing", "Rescattering2", "SigmaPole", "SympyLineshape",
    "Resonance", "ResonanceAmplitude", "ResonanceContext", "SCFSignalPDF", "SignalPDF",
    "SparseMigration", "SquareDalitzGrid", "SquareDalitzHistogramBackground", "SquareDalitzHistogramEfficiency", "SumPDF1D",
    "SquareDalitzSCFMap", "ToyBackground", "VetoMap", "VetoedDensity", "WeightedUnbinnedNLL", "ZemachP", "ZemachPstar",
    "Zemach_P", "Zemach_Pstar",
    "boost_to_rest_frame", "binned_data", "chi2_from_histograms", "covariant_kinematics",
    "covariant_kinematics_from_invariants", "dalitz_s13_limits",
    "cp_models_from_spec", "cp_models_to_spec", "cp_models_with_fitted_values",
    "delta_method_covariance", "delta_method_errors", "delta_method_jacobian",
    "enable_x64", "export_cp_models", "export_model",
    "fold_thetaprime",
    "generate_cp_toy", "generate_signal_toy", "generate_toy", "histogram_background_from_root",
    "histogram_efficiency_from_root", "import_cp_models", "import_model", "invariants_to_square_dalitz",
    "model_from_spec", "model_to_spec", "model_with_fitted_values", "physical_bin_mask",
    "plot_binned_data", "plot_contour", "plot_dalitz", "plot_pulls", "plot_square_dalitz", "point_to_point_dissimilarity", "prepare_inverse_toy_generator",
    "read_phase_space_sample", "read_root_histogram2d", "read_root_tree",
    "square_dalitz_background_from_root", "square_dalitz_efficiency_from_root", "square_dalitz_jacobian",
    "square_dalitz_to_invariants", "vetoed_signal_pdf", "weighted_resample",
    "write_cp_phase_space_sample", "write_phase_space_sample", "write_phase_space_samples",
    "YieldAsymmetry",
]
