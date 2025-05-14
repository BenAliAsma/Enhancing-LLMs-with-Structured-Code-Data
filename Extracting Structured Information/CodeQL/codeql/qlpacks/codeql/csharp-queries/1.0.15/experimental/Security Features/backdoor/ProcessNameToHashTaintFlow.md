# ProcessName to hash function flow
This query detects code flow from ProcessName property on the Process class into a hash function.

Such flow is often used in code backdoors to detect running processes and compare them to an obfuscated list of antivirus processes to avoid detection. Solorigate is one example that uses this mechanism.


## Recommendation
Any findings from this rule are only intended to indicate suspicious code that shares similarities with known portions of code used for the Solorigate attack. There is no certainty that the code is related or that the code is part of any attack.

For more information about Solorigate, please visit https://aka.ms/solorigate.

