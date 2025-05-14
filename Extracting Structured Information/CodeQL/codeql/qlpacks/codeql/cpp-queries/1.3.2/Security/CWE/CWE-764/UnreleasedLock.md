# Lock may not be released
When a thread acquires a lock it must make sure to unlock it again; failing to do so can lead to deadlocks. If a lock allows a thread to acquire it multiple times, for example `std::recursive_mutex`, then the number of locks must match the number of unlocks in order to fully release the lock.


## Recommendation
The best way to ensure that locks are always unlocked is to use RAII (Resource Acquisition Is Initialization). That means acquiring the lock in the constructor of a class, and releasing it in its destructor. A local-scoped instance of that class will then be destroyed when it leaves scope, even if an exception is thrown, ensuring that the lock is released.


## Example
In this example, the mutex may not be unlocked if the function returns early.


```cpp
std::mutex mutex;
int fun() {
	mutex.lock();
	bool succeeded = doWork();
	if (!succeeded) {
		// BAD: this does not release the mutex
		return -1
	}
	mutex.unlock();
	return 1;
}

```
In this second example, we show a simple RAII wrapper class for `std::mutex`. Using this ensures that even in the case of the early return, the mutex is released.


```cpp
class RAII_Mutex
{
	std::mutex lock;
public:
	RAII_Mutex(mutex m) : lock(m)
	{
		lock.lock();
	}

	~RAII_Mutex()
	{
		lock.unlock();
	}
};


std::mutex mutex;
int fun() {
	RAII_Mutex(mutex);

	bool succeeded = doWork();
	if (!succeeded) {
		// GOOD: the RAII_Mutex is destroyed, releasing the lock
		return -1
	}
	
	return 1;
}
```

## References
* Common Weakness Enumeration: [CWE-764](https://cwe.mitre.org/data/definitions/764.html).
* Common Weakness Enumeration: [CWE-833](https://cwe.mitre.org/data/definitions/833.html).
