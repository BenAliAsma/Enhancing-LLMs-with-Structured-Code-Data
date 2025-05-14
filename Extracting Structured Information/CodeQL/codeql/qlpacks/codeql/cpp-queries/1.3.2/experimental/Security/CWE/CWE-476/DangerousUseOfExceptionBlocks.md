# Dangerous use of exception blocks.
When releasing memory in a catch block, be sure that the memory was allocated and has not already been released.


## Example
The following example shows erroneous and fixed ways to use exception handling.


```cpp
...
try {
  if (checkValue) throw exception();
  bufMyData =  new myData*[sizeInt];
	 
  } 
  catch (...) 
  {
    for (size_t i = 0; i < sizeInt; i++)
    {
    	delete[] bufMyData[i]->buffer; // BAD
        delete bufMyData[i];
    }
...
try {
  if (checkValue) throw exception();
  bufMyData =  new myData*[sizeInt];
	 
  } 
  catch (...) 
  {
    for (size_t i = 0; i < sizeInt; i++)
    {
      if(bufMyData[i])
      {
    	delete[] bufMyData[i]->buffer; // GOOD
        delete bufMyData[i];
      }
    }

...
  catch (const exception &) {
	  delete valData;
	  throw;
  }
  catch (...) 
  {
    delete valData; // BAD
...
  catch (const exception &) {
	  delete valData;
    valData = NULL;
	  throw;
  }
  catch (...) 
  {
    delete valData; // GOOD  
...

```

## References
* CERT C Coding Standard: [EXP34-C. Do not dereference null pointers](https://wiki.sei.cmu.edu/confluence/display/c/EXP34-C.+Do+not+dereference+null+pointers).
* Common Weakness Enumeration: [CWE-476](https://cwe.mitre.org/data/definitions/476.html).
* Common Weakness Enumeration: [CWE-415](https://cwe.mitre.org/data/definitions/415.html).
