# Security sensitive JsonWebTokenHandler validations are disabled
Token validation checks ensure that while validating tokens, all aspects are analyzed and verified. Turning off validation can lead to security holes by allowing untrusted tokens to make it through validation.


## Recommendation
Set `Microsoft.IdentityModel.Tokens.TokenValidationParameters` properties `RequireExpirationTime`, `ValidateAudience`, `ValidateIssuer`, or `ValidateLifetime` to `true`. Or, remove the assignment to `false` because the default value is `true`.


## Example
This example disabled the validation.


```csharp
using System;
using Microsoft.IdentityModel.Tokens;
class TestClass
{
    public void TestMethod()
    {
        TokenValidationParameters parameters = new TokenValidationParameters();
        parameters.RequireExpirationTime = false;
        parameters.ValidateAudience = false;
        parameters.ValidateIssuer = false;
        parameters.ValidateLifetime = false;
    }
}
```
To fix it, do not disable the validations or use the default value.


```csharp
using System;
using Microsoft.IdentityModel.Tokens;
class TestClass
{
    public void TestMethod()
    {
        TokenValidationParameters parameters = new TokenValidationParameters();
        parameters.RequireExpirationTime = true;
        parameters.ValidateAudience = true;
        parameters.ValidateIssuer = true;
        parameters.ValidateLifetime = true;
    }
}
```

## References
* [azure-activedirectory-identitymodel-extensions-for-dotnet ValidatingTokens wiki](https://aka.ms/wilson/tokenvalidation)
