# Code Snippets for Entity: Linear1D (class)

**Project:** astropy/astropy
**Commit:** fa4e8d1cd279acf9b24560813c8652494ccd5922
**Version:** 5.1
**Date:** 2023-02-06T21:56:51Z

**Problem Statement:**

Modeling's `separability_matrix` does not compute separability correctly for nested CompoundModels
Consider the following model:

```python
from astropy.modeling import models as m
from astropy.modeling.separable import separability_matrix

cm = m.Linear1D(10) & m.Linear1D(5)
```

It's separability matrix as you might expect is a diagonal:

```python
>>> separability_matrix(cm)
array([[ True, False],
[False, True]])
```

If I make the model more complex:
```python
>>> separability_matrix(m.Pix2Sky_TAN() & m.Linear1D(10) & m.Linear1D(5))
array([[ True, True, False, False],
[ True, True, False, False],
[False, False, True, False],
[False, False, False, True]])
```

The output matrix is again, as expected, the outputs and inputs to the linear models are separable and independent of each other.

If however, I nest these compound models:
```python
>>> separability_matrix(m.Pix2Sky_TAN() & cm)
array([[ True, True, False, False],
[ True, True, False, False],
[False, False, True, True],
[False, False, True, True]])
```
Suddenly the inputs and outputs are no longer separable?

This feels like a bug to me, but I might be missing something?


---

## Snippet 1 — `astropy/modeling/functional_models.py` — Range: [1326, 0, 1380, 85]

**File:** `astropy/modeling/functional_models.py`
**Range:** Lines 1326-1380, Chars 0-85

```python
        if isinstance(argument, Quantity):
            argument = argument.value
        arc_cos = np.arccos(argument) / TWOPI

        return (arc_cos - phase) / frequency

    @staticmethod
    def fit_deriv(x, amplitude, frequency, phase):
        """One dimensional ArcCosine model derivative."""
        d_amplitude = x / (
            TWOPI * frequency * amplitude**2 * np.sqrt(1 - (x / amplitude) ** 2)
        )
        d_frequency = (phase - (np.arccos(x / amplitude) / TWOPI)) / frequency**2
        d_phase = -1 / frequency * np.ones(x.shape)
        return [d_amplitude, d_frequency, d_phase]

    def bounding_box(self):
        """
        Tuple defining the default ``bounding_box`` limits,
        ``(x_low, x_high)``.
        """
        return -1 * self.amplitude, 1 * self.amplitude

    @property
    def inverse(self):
        """One dimensional inverse of ArcCosine."""
        return Cosine1D(
            amplitude=self.amplitude, frequency=self.frequency, phase=self.phase
        )


class ArcTangent1D(_InverseTrigonometric1D):
    """
    One dimensional ArcTangent model returning values between -pi/2 and
    pi/2 only.

    Parameters
    ----------
    amplitude : float
        Oscillation amplitude for corresponding Tangent
    frequency : float
        Oscillation frequency for corresponding Tangent
    phase : float
        Oscillation phase for corresponding Tangent

    See Also
    --------
    Tangent1D, ArcSine1D, ArcCosine1D


    Notes
    -----
    Model formula:

        .. math:: f(x) = ((arctan(x / A) / 2pi) - p) / f
```

## Snippet 2 — `astropy/modeling/functional_models.py` — Range: [1352, 4, 1356, 36]

**File:** `astropy/modeling/functional_models.py`
**Range:** Lines 1352-1356, Chars 4-36

```python
    return Cosine1D(
            amplitude=self.amplitude, frequency=self.frequency, phase=self.phase
        )
```

## Snippet 3 — `astropy/modeling/functional_models.py` — Range: [1358, 4, 1364, 37]

**File:** `astropy/modeling/functional_models.py`
**Range:** Lines 1358-1364, Chars 4-37

```python
"""
    One dimensional ArcTangent model returning values between -pi/2 and
    pi/2 only.

    Parameters
    ----------
    amplitude : float
```

## Snippet 4 — `astropy/modeling/functional_models.py` — Range: [1366, 4, 1370, 71]

**File:** `astropy/modeling/functional_models.py`
**Range:** Lines 1366-1370, Chars 4-71

```python
frequency : float
        Oscillation frequency for corresponding Tangent
    phase : float
        Oscillation phase for corresponding Tangent
```

## Snippet 5 — `astropy/modeling/functional_models.py` — Range: [1372, 4, 1376, 70]

**File:** `astropy/modeling/functional_models.py`
**Range:** Lines 1372-1376, Chars 4-70

```python
--------
    Tangent1D, ArcSine1D, ArcCosine1D


    Notes
```

## Snippet 6 — `astropy/modeling/functional_models.py` — Range: [1378, 4, 1380, 85]

**File:** `astropy/modeling/functional_models.py`
**Range:** Lines 1378-1380, Chars 4-85

```python
Model formula:

        .. math:: f(x) = ((arctan(x / A) / 2pi) - p) / f
```

