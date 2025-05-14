# Erroneous class compare
Checking the type of an object on its type's name is not robust to future changes, and may allow an attacker to inject a malicious class.


## Recommendation
Implement a check of the object's type that is not solely based on its class name.


## Example
In this example the type of the account is checked purely based on class name and is not reliable.

It would be more appropriate to check the type of `account` like so:


```csharp
class ErroneousClassCompareFix
{
    public static void ApproveTransaction(object account, Transaction transaction)
    {
        if (account.GetType() == typeof(Trusted.Bank.Account))
        {
            transaction.Process();
        }
    }
}

```
