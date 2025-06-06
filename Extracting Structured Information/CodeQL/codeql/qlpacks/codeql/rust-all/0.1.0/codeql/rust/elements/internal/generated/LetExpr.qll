// generated by codegen, do not edit
/**
 * This module provides the generated definition of `LetExpr`.
 * INTERNAL: Do not import directly.
 */

private import codeql.rust.elements.internal.generated.Synth
private import codeql.rust.elements.internal.generated.Raw
import codeql.rust.elements.Attr
import codeql.rust.elements.Expr
import codeql.rust.elements.internal.ExprImpl::Impl as ExprImpl
import codeql.rust.elements.Pat

/**
 * INTERNAL: This module contains the fully generated definition of `LetExpr` and should not
 * be referenced directly.
 */
module Generated {
  /**
   * A `let` expression. For example:
   * ```rust
   * if let Some(x) = maybe_some {
   *     println!("{}", x);
   * }
   * ```
   * INTERNAL: Do not reference the `Generated::LetExpr` class directly.
   * Use the subclass `LetExpr`, where the following predicates are available.
   */
  class LetExpr extends Synth::TLetExpr, ExprImpl::Expr {
    override string getAPrimaryQlClass() { result = "LetExpr" }

    /**
     * Gets the `index`th attr of this let expression (0-based).
     */
    Attr getAttr(int index) {
      result =
        Synth::convertAttrFromRaw(Synth::convertLetExprToRaw(this).(Raw::LetExpr).getAttr(index))
    }

    /**
     * Gets any of the attrs of this let expression.
     */
    final Attr getAnAttr() { result = this.getAttr(_) }

    /**
     * Gets the number of attrs of this let expression.
     */
    final int getNumberOfAttrs() { result = count(int i | exists(this.getAttr(i))) }

    /**
     * Gets the scrutinee of this let expression, if it exists.
     */
    Expr getScrutinee() {
      result =
        Synth::convertExprFromRaw(Synth::convertLetExprToRaw(this).(Raw::LetExpr).getScrutinee())
    }

    /**
     * Holds if `getScrutinee()` exists.
     */
    final predicate hasScrutinee() { exists(this.getScrutinee()) }

    /**
     * Gets the pattern of this let expression, if it exists.
     */
    Pat getPat() {
      result = Synth::convertPatFromRaw(Synth::convertLetExprToRaw(this).(Raw::LetExpr).getPat())
    }

    /**
     * Holds if `getPat()` exists.
     */
    final predicate hasPat() { exists(this.getPat()) }
  }
}
