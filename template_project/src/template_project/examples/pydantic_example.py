import pydantic as pdt

class GridCVSearcher(pdt.BaseModel):
    """Grid searcher with cross-fold validation for better model performance metrics."""

    n_jobs: int | None = None  # Number of parallel jobs, None means 1
    refit: bool = True  # Refit the best estimator
    verbose: int = 3  # Level of verbosity
    error_score: str | float = "raise"  # 'raise' or a float 
    return_train_score: bool = False  # Return training scores

    # Validator for `verbose` to ensure non-negative values
    @pdt.field_validator("verbose")
    def validate_verbose(cls, v):
        if v < 0:
            raise ValueError("`verbose` must be non-negative.")
        return v

    # Validator to ensure `error_score` is valid
    @pdt.field_validator("error_score")
    def validate_error_score(cls, v):
        if isinstance(v, float) and v < 0:
            raise ValueError("`error_score` must be 'raise' or a non-negative float.")
        return v

    class Config:
        validate_assignment = True  # Enable re-validation on assignment


# Example usage of GridCVSearcher
try:
    # Valid input
    config = GridCVSearcher(
        n_jobs=4,
        refit=False,
        verbose=2,
        error_score=0.5,
        return_train_score=True,
    )
    print("Valid Configuration:", config)

    # Accessing and modifying attributes with re-validation
    config.verbose = 1  # This works
    print("Valid Configuration:", config)

    # Invalid assignment (will raise ValidationError)
    config.error_score = "test"  # This will raise a ValueError
    print("Valid Configuration:", config)
except pdt.ValidationError as e:
    print("Validation Error:", e)

