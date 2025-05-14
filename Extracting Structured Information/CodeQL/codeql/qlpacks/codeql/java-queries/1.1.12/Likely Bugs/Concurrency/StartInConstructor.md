# Start of thread in constructor
Starting a thread within a constructor may cause unexpected results. If the class is extended, the thread may start before the subclass constructor has completed its initialization, which may not be intended.


## Recommendation
Avoid starting threads in constructors. Typically, the constructor of a class only *constructs* the thread object, and a separate `start` method should be provided to *start* the thread object created by the constructor.


## Example
In the following example, because the `Test` constructor implicitly calls the `Super` constructor, the thread created in the `Super` constructor may start before `this.name` has been initialized. Therefore, the program may output "hello " followed by a null string.


```java
class Super {
    public Super() {
        new Thread() {
            public void run() {
                System.out.println(Super.this.toString());
            }
        }.start(); // BAD: The thread is started in the constructor of 'Super'.
    }

    public String toString() {
        return "hello";
    }
}

class Test extends Super {
    private String name;
    public Test(String nm) {
        // The thread is started before
        // this line is run
        this.name = nm;
    }

    public String toString() {
        return super.toString() + " " + name;
    }

    public static void main(String[] args) {
        new Test("my friend");
    }
}
```
In the following modified example, the thread created in the `Super` constructor is not started within the constructor; `main` starts the thread after `this.name` has been initialized. This results in the program outputting "hello my friend".


```java
class Super {
    Thread thread;
    public Super() {
        thread = new Thread() {
            public void run() {
                System.out.println(Super.this.toString());
            }
        };
    }

    public void start() {  // good
        thread.start();
    }
    
    public String toString() {
        return "hello";
    }
}

class Test extends Super {
    private String name;
    public Test(String nm) {
        this.name = nm;
    }

    public String toString() {
        return super.toString() + " " + name;
    }

    public static void main(String[] args) {
        Test t = new Test("my friend");
        t.start();
    }
}
```

## References
* IBM developerWorks: [Don't start threads from within constructors](https://web.archive.org/web/20200417101823/http://www.ibm.com/developerworks/java/library/j-jtp0618/index.html#4).
