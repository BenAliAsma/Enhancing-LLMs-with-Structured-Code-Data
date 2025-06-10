# All Code Snippets - Consolidated Report

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

## Entity Rankings

Entities are ranked by number of code blocks (relevance indicator):

1. **Linear1D (class)** (103 blocks)
2. **astropy.modeling.separable (path)** (20 blocks)
3. **separability_matrix (function)** (18 blocks)

---

# Rank 1: Linear1D (class)

**Relevance Score:** 103 blocks

## Linear1D (class) - Snippet 1

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

## Linear1D (class) - Snippet 2

**File:** `astropy/modeling/functional_models.py`
**Range:** Lines 1352-1356, Chars 4-36

```python
    return Cosine1D(
            amplitude=self.amplitude, frequency=self.frequency, phase=self.phase
        )
```

## Linear1D (class) - Snippet 3

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

## Linear1D (class) - Snippet 4

**File:** `astropy/modeling/functional_models.py`
**Range:** Lines 1366-1370, Chars 4-71

```python
frequency : float
        Oscillation frequency for corresponding Tangent
    phase : float
        Oscillation phase for corresponding Tangent
```

## Linear1D (class) - Snippet 5

**File:** `astropy/modeling/functional_models.py`
**Range:** Lines 1372-1376, Chars 4-70

```python
--------
    Tangent1D, ArcSine1D, ArcCosine1D


    Notes
```

## Linear1D (class) - Snippet 6

**File:** `astropy/modeling/functional_models.py`
**Range:** Lines 1378-1380, Chars 4-85

```python
Model formula:

        .. math:: f(x) = ((arctan(x / A) / 2pi) - p) / f
```

---

# Rank 2: astropy.modeling.separable (path)

**Relevance Score:** 20 blocks

## astropy.modeling.separable (path) - Snippet 1

**File:** `astropy/modeling/separable.py`
**Range:** Lines 26-62, Chars 0-23

```python
    """
    A separability test for the outputs of a transform.

    Parameters
    ----------
    transform : `~astropy.modeling.core.Model`
        A (compound) model.

    Returns
    -------
    is_separable : ndarray
        A boolean array with size ``transform.n_outputs`` where
        each element indicates whether the output is independent
        and the result of a separable transform.

    Examples
    --------
    >>> from astropy.modeling.models import Shift, Scale, Rotation2D, Polynomial2D
    >>> is_separable(Shift(1) & Shift(2) | Scale(1) & Scale(2))
        array([ True,  True]...)
    >>> is_separable(Shift(1) & Shift(2) | Rotation2D(2))
        array([False, False]...)
    >>> is_separable(Shift(1) & Shift(2) | Mapping([0, 1, 0, 1]) | \
        Polynomial2D(1) & Polynomial2D(2))
        array([False, False]...)
    >>> is_separable(Shift(1) & Shift(2) | Mapping([0, 1, 0, 1]))
        array([ True,  True,  True,  True]...)

    """
    if transform.n_inputs == 1 and transform.n_outputs > 1:
        is_separable = np.array([False] * transform.n_outputs).T
        return is_separable
    separable_matrix = _separable(transform)
    is_separable = separable_matrix.sum(1)
    is_separable = np.where(is_separable != 1, False, True)
    return is_separable
```

## astropy.modeling.separable (path) - Snippet 2

**File:** `astropy/modeling/separable.py`
**Range:** Lines 65-101, Chars 0-27

```python
    """
    Compute the correlation between outputs and inputs.

    Parameters
    ----------
    transform : `~astropy.modeling.core.Model`
        A (compound) model.

    Returns
    -------
    separable_matrix : ndarray
        A boolean correlation matrix of shape (n_outputs, n_inputs).
        Indicates the dependence of outputs on inputs. For completely
        independent outputs, the diagonal elements are True and
        off-diagonal elements are False.

    Examples
    --------
    >>> from astropy.modeling.models import Shift, Scale, Rotation2D, Polynomial2D
    >>> separability_matrix(Shift(1) & Shift(2) | Scale(1) & Scale(2))
        array([[ True, False], [False,  True]]...)
    >>> separability_matrix(Shift(1) & Shift(2) | Rotation2D(2))
        array([[ True,  True], [ True,  True]]...)
    >>> separability_matrix(Shift(1) & Shift(2) | Mapping([0, 1, 0, 1]) | \
        Polynomial2D(1) & Polynomial2D(2))
        array([[ True,  True], [ True,  True]]...)
    >>> separability_matrix(Shift(1) & Shift(2) | Mapping([0, 1, 0, 1]))
        array([[ True, False], [False,  True], [ True, False], [False,  True]]...)

    """
    if transform.n_inputs == 1 and transform.n_outputs > 1:
        return np.ones((transform.n_outputs, transform.n_inputs), dtype=np.bool_)
    separable_matrix = _separable(transform)
    separable_matrix = np.where(separable_matrix != 0, True, False)
    return separable_matrix
```

## astropy.modeling.separable (path) - Snippet 3

**File:** `astropy/modeling/separable.py`
**Range:** Lines 104-126, Chars 0-16

```python
    Compute the number of outputs of two models.

    The two models are the left and right model to an operation in
    the expression tree of a compound model.

    Parameters
    ----------
    left, right : `astropy.modeling.Model` or ndarray
        If input is of an array, it is the output of `coord_matrix`.

    """
    if isinstance(left, Model):
        lnout = left.n_outputs
    else:
        lnout = left.shape[0]
    if isinstance(right, Model):
        rnout = right.n_outputs
    else:
        rnout = right.shape[0]
    noutp = lnout + rnout
    return noutp
```

---

# Rank 3: separability_matrix (function)

**Relevance Score:** 18 blocks

## separability_matrix (function) - Snippet 1

**File:** `astropy/modeling/separable.py`
**Range:** Lines 65-101, Chars 0-27

```python
    """
    Compute the correlation between outputs and inputs.

    Parameters
    ----------
    transform : `~astropy.modeling.core.Model`
        A (compound) model.

    Returns
    -------
    separable_matrix : ndarray
        A boolean correlation matrix of shape (n_outputs, n_inputs).
        Indicates the dependence of outputs on inputs. For completely
        independent outputs, the diagonal elements are True and
        off-diagonal elements are False.

    Examples
    --------
    >>> from astropy.modeling.models import Shift, Scale, Rotation2D, Polynomial2D
    >>> separability_matrix(Shift(1) & Shift(2) | Scale(1) & Scale(2))
        array([[ True, False], [False,  True]]...)
    >>> separability_matrix(Shift(1) & Shift(2) | Rotation2D(2))
        array([[ True,  True], [ True,  True]]...)
    >>> separability_matrix(Shift(1) & Shift(2) | Mapping([0, 1, 0, 1]) | \
        Polynomial2D(1) & Polynomial2D(2))
        array([[ True,  True], [ True,  True]]...)
    >>> separability_matrix(Shift(1) & Shift(2) | Mapping([0, 1, 0, 1]))
        array([[ True, False], [False,  True], [ True, False], [False,  True]]...)

    """
    if transform.n_inputs == 1 and transform.n_outputs > 1:
        return np.ones((transform.n_outputs, transform.n_inputs), dtype=np.bool_)
    separable_matrix = _separable(transform)
    separable_matrix = np.where(separable_matrix != 0, True, False)
    return separable_matrix
```

---


**Summary:** 3 entities, 10 total snippets
