# Resource not released in destructor
This rule finds resources that are allocated by a class, but not released in the destructor of that class. Allocating a resource includes:

* Allocating memory with `malloc`
* Creating objects using `new`
* Opening files
* Opening network sockets
Resource management can be a complex task, so a standard best practice is the pattern of *Resource Acquisition is Initialization* (RAII). In an RAII class, the constructor allocates all required resources, and the destructor frees all resources. This guarantees that simply deleting an instance of the class is enough to free resources, and benefits from C++'s automatic object lifetime management. A well-designed RAII class cannot be a source of resource leaks as long as its lifetime is properly managed:

* If it is allocated with `new` it should be released with `delete` by the client that created it
* If its lifetime is lexical (it is only used in one function), then it is enough to declare it as a local variable of that function (not a pointer) and the C++ runtime will ensure it is released on exit.
There are two possible messages:

* "Resource *x* is acquired by class *C* but not released in the destructor. It is released from *f*, so this function may need to be called in the destructor".
This indicates that the resource (*x*) is being released, just not in the destructor of the class. Typically, it is released in a function called `close`, `free` or something similar. This does not always indicate a resource leak, but it shows that the class is unnecessarily difficult to use because it does not conform to the RAII pattern.

* "Resource *x* is acquired by class *C* but not released anywhere in this class".
This indicates that the class is allocating resources but not responsible for releasing them. This is very error-prone: even if a class requires an explicit close operator, it should manage any resources it allocates rather than forcing clients to manage them. In the worst case this can be a resource leak, if client code does not free the resource.


## Recommendation
If the resource is not being released at all, ensure that the class does release the resource, normally by adding the release to the destructor of the class. This change needs to be carefully validated: client code may be relying on the resource outliving the class that allocated it, and must be reviewed and updated if necessary.

In the other case, for instance a class that has an explicit `close` function, the aim is to migrate the class to a straightforward RAII pattern. This can be achieved in several steps:

1. First, ensure that the `close` function (or its equivalent) is safe to call twice, by releasing resources only if they have not been released before.
1. Next, call the `close` function from the destructor. This does not require changing the client code, since it is safe to call it twice.
1. Migrate client code to remove direct uses of the `close` function, taking the opportunity to check that the object itself is being deleted appropriately.
1. Finally, when possible also migrate initialization code to the constructor of the class to make it follow the RAII pattern precisely.

## Example

```cpp
// This class opens a file but never closes it. Even its clients
// cannot close the file
class ResourceLeak {
private:
    int sockfd;
    FILE* file;
public:
    C() {
        sockfd = socket(AF_INET, SOCK_STREAM, 0);
    }

    void f() {
        file = fopen("foo.txt", "r");
        ...
    }
};

// This class relies on its client to release any stream it
// allocates. Note that this means the client must have
// intimate knowledge of the implementation of the class to
// decide whether it is safe to release the stream. 
class StreamPool {
private:
  Stream *instance;
public:
  Stream *createStream(char *name) {
    if (!instance) 
      instance = new Stream(name);
    return instance;
  }
}

// This class handles its resources, but does not do that in
// the constructor/destructor. It can be rewritten easily to
// be safer to use.
class StreamHandler {
private:
  char *_name;
  Stream *stream;
public:
  C(char *name) {
    _name = strdup(name):
  }
  void open() {
    stream = new Stream();
  }
  void close() {
    delete stream;
  }
  ~StreamHandler() {
    free(_name);
    // stream should be deleted here, not in close()
  }
}
```

## References
* AV Rule 79, *Joint Strike Fighter Air Vehicle C++ Coding Standards*. Lockheed Martin Corporation, 2005.
* S. Meyers. *Effective C++ 3d ed.* pp 61-66. Addison-Wesley Professional, 2005.
* [Resource Acquisition Is Initialization](http://en.wikipedia.org/wiki/Resource_Acquisition_Is_Initialization)
* Common Weakness Enumeration: [CWE-404](https://cwe.mitre.org/data/definitions/404.html).
