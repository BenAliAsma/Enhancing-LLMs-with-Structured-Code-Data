# Insecure Bean Validation
Custom error messages for constraint validators support different types of interpolation, including [Java EL expressions](https://docs.jboss.org/hibernate/validator/5.1/reference/en-US/html/chapter-message-interpolation.html#section-interpolation-with-message-expressions). Controlling part of the message template being passed to `ConstraintValidatorContext.buildConstraintViolationWithTemplate()` argument can lead to arbitrary Java code execution. Unfortunately, it is common that validated (and therefore, normally untrusted) bean properties flow into the custom error message.


## Recommendation
There are different approaches to remediate the issue:

* Do not include validated bean properties in the custom error message.
* Use parameterized messages instead of string concatenation. For example:
```
HibernateConstraintValidatorContext context =
   constraintValidatorContext.unwrap(HibernateConstraintValidatorContext.class);
context.addMessageParameter("foo", "bar");
context.buildConstraintViolationWithTemplate("My violation message contains a parameter {foo}")
   .addConstraintViolation();
```
* Sanitize the validated bean properties to make sure that there are no EL expressions. An example of valid sanitization logic can be found [here](https://github.com/hibernate/hibernate-validator/blob/master/engine/src/main/java/org/hibernate/validator/internal/engine/messageinterpolation/util/InterpolationHelper.java#L17).
* Disable the EL interpolation and only use `ParameterMessageInterpolator`:
```
Validator validator = Validation.byDefaultProvider()
   .configure()
   .messageInterpolator(new ParameterMessageInterpolator())
   .buildValidatorFactory()
   .getValidator();
```
* Replace Hibernate Validator with Apache BVal, which in its latest version does not interpolate EL expressions by default. Note that this replacement may not be a simple drop-in replacement.

## Example
The following validator could result in arbitrary Java code execution:


```java
import javax.validation.ConstraintValidator;
import javax.validation.ConstraintValidatorContext;
import org.hibernate.validator.constraintvalidation.HibernateConstraintValidatorContext;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class TestValidator implements ConstraintValidator<Object, String> {

    public static class InterpolationHelper {

        public static final char BEGIN_TERM = '{';
        public static final char END_TERM = '}';
        public static final char EL_DESIGNATOR = '$';
        public static final char ESCAPE_CHARACTER = '\\';

        private static final Pattern ESCAPE_MESSAGE_PARAMETER_PATTERN = Pattern.compile( "([\\" + ESCAPE_CHARACTER + BEGIN_TERM + END_TERM + EL_DESIGNATOR + "])" );

        private InterpolationHelper() {
        }

        public static String escapeMessageParameter(String messageParameter) {
            if ( messageParameter == null ) {
                return null;
            }
            return ESCAPE_MESSAGE_PARAMETER_PATTERN.matcher( messageParameter ).replaceAll( Matcher.quoteReplacement( String.valueOf( ESCAPE_CHARACTER ) ) + "$1" );
        }

    }

    @Override
    public boolean isValid(String object, ConstraintValidatorContext constraintContext) {
        String value = object + " is invalid";

        // Bad: Bean properties (normally user-controlled) are passed directly to `buildConstraintViolationWithTemplate`
        constraintContext.buildConstraintViolationWithTemplate(value).addConstraintViolation().disableDefaultConstraintViolation();

        // Good: Bean properties (normally user-controlled) are escaped 
        String escaped = InterpolationHelper.escapeMessageParameter(value);
        constraintContext.buildConstraintViolationWithTemplate(escaped).addConstraintViolation().disableDefaultConstraintViolation();

        // Good: Bean properties (normally user-controlled) are parameterized
        HibernateConstraintValidatorContext context = constraintContext.unwrap( HibernateConstraintValidatorContext.class );
        context.addMessageParameter( "prop", object );
        context.buildConstraintViolationWithTemplate( "{prop} is invalid").addConstraintViolation();
        return false;
    }

}

```

## References
* Hibernate Reference Guide: [ConstraintValidatorContext](https://docs.jboss.org/hibernate/stable/validator/reference/en-US/html_single/#_the_code_constraintvalidatorcontext_code).
* GitHub Security Lab research: [Bean validation](https://securitylab.github.com/research/bean-validation-RCE).
* Common Weakness Enumeration: [CWE-94](https://cwe.mitre.org/data/definitions/94.html).
