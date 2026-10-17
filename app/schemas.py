"""Validated request/response schemas for the prediction API."""
from pydantic import BaseModel, ConfigDict, Field


class WineFeatures(BaseModel):
    """Numeric chemical measurements for a single wine sample."""
    model_config = ConfigDict(extra="forbid")

    alcohol: float = Field(gt=0, allow_inf_nan=False, description="Alcohol content")
    malic_acid: float = Field(gt=0, allow_inf_nan=False, description="Malic acid")
    ash: float = Field(gt=0, allow_inf_nan=False, description="Ash")
    alcalinity_of_ash: float = Field(gt=0, allow_inf_nan=False, description="Alcalinity of ash")
    magnesium: float = Field(gt=0, allow_inf_nan=False, description="Magnesium")
    total_phenols: float = Field(gt=0, allow_inf_nan=False, description="Total phenols")
    flavanoids: float = Field(gt=0, allow_inf_nan=False, description="Flavanoids")
    nonflavanoid_phenols: float = Field(gt=0, allow_inf_nan=False, description="Nonflavanoid phenols")
    proanthocyanins: float = Field(gt=0, allow_inf_nan=False, description="Proanthocyanins")
    color_intensity: float = Field(gt=0, allow_inf_nan=False, description="Color intensity")
    hue: float = Field(gt=0, allow_inf_nan=False, description="Hue")
    od280_od315_of_diluted_wines: float = Field(gt=0, allow_inf_nan=False, description="OD280/OD315 of diluted wines")
    proline: float = Field(gt=0, allow_inf_nan=False, description="Proline")


class PredictionResponse(BaseModel):
    predicted_class: int
    class_label: str
    probabilities: dict[str, float]
    model_name: str
