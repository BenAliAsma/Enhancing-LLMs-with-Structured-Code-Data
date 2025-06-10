"""
Code Snippets for Entity: astropy.modeling.separable (path)
Generated from project: astropy/astropy
Commit: fa4e8d1cd279acf9b24560813c8652494ccd5922
Version: 5.1
Date: 2023-02-06T21:56:51Z

Problem Statement:

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

"""

# Common imports (uncomment as needed)
# import numpy as np
# import matplotlib.pyplot as plt
# from astropy.modeling import models as m
# from astropy.modeling.separable import separability_matrix

# ======================================================================
# SNIPPET 1: astropy/modeling/separable.py (Lines 26-62)
# ======================================================================

def snippet_1_astropy_modeling_separable():
    """
    Snippet from astropy/modeling/separable.py (Lines 26-62)
    """
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

# ======================================================================
# SNIPPET 2: astropy/modeling/separable.py (Lines 65-101)
# ======================================================================

def snippet_2_astropy_modeling_separable():
    """
    Snippet from astropy/modeling/separable.py (Lines 65-101)
    """
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

# ======================================================================
# SNIPPET 3: astropy/modeling/separable.py (Lines 104-126)
# ======================================================================

def snippet_3_astropy_modeling_separable():
    """
    Snippet from astropy/modeling/separable.py (Lines 104-126)
    """
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


======================================================================
# USAGE EXAMPLES
======================================================================

if __name__ == '__main__':
    print(f'Code snippets for entity: {entity}')
    print(f'Total snippets: {len(snippets)}')
    
    # Uncomment to run specific snippets:
    # snippet_1_astropy_modeling_separable()
    # snippet_2_astropy_modeling_separable()
    # snippet_3_astropy_modeling_separable()
