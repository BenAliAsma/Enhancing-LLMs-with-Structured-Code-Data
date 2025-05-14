# Potential Timebomb
This query detects situations in which an offset to a last file modification time is used to conditionally execute a particular block of code. This is a common pattern in backdoors, where the file's modification timestamp is the time at which the backdoor was planted, and the time offset is used as a time bomb before a particular code block is executed.


## Recommendation
Any findings from this rule are only intended to indicate suspicious code that shares similarities with known portions of code used for the Solorigate attack. There is no certainty that the code is related or that the code is part of any attack.

For more information about Solorigate, please visit https://aka.ms/solorigate.

