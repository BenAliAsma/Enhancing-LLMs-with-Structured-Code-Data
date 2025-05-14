# Delegated security sensitive validations for JsonWebTokenHandler always return true, medium precision
By setting critical `TokenValidationParameter` validation delegates to always return `true`, important authentication safeguards are disabled. Disabling safeguards can lead to incorrect validation of tokens from any issuer or expired tokens.


## Recommendation
Improve the logic of the delegate so not all code paths return `true`, which effectively disables that type of validation; or throw `SecurityTokenInvalidAudienceException` or `SecurityTokenInvalidLifetimeException` in failure cases when you want to fail validation and have other cases pass by returning `true`.


## Example
This example delegates `AudienceValidator` to a callable that always returns true.


```csharp
using System;
using Microsoft.IdentityModel.Tokens;
class TestClass
{
    public void TestMethod()
    {
        TokenValidationParameters parameters = new TokenValidationParameters();
        parameters.AudienceValidator = (audiences, token, tvp) => { return true; };
    }
}
```
To fix it, use a callable that performs a validation, and fails when appropriate.


```csharp
using System;
using Microsoft.IdentityModel.Tokens;
class TestClass
{
    public void TestMethod()
    {
        TokenValidationParameters parameters = new TokenValidationParameters();
        parameters.AudienceValidator = (audiences, token, tvp) =>
        {
            // Implement your own custom audience validation
            if (PerformCustomAudienceValidation(audiences, token))
                return true;
            else
                return false;
        };
    }
}
```

## References
* [azure-activedirectory-identitymodel-extensions-for-dotnet ValidatingTokens wiki](https://aka.ms/wilson/tokenvalidation)
