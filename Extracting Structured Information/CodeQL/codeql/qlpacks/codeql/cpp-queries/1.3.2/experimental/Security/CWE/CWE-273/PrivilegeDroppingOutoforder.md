# LinuxPrivilegeDroppingOutoforder
The code attempts to drop privilege in an incorrect order by erroneous dropping user privilege before groups. This has security impact if the return codes are not checked.

False positives include code performing negative checks, making sure that setgid or setgroups does not work, meaning permissions are dropped. Additionally, other forms of sandboxing may be present removing any residual risk, for example a dedicated user namespace.


## Recommendation
Set the new group ID, then set the target user's intended groups by dropping previous supplemental source groups and initializing target groups, and finally set the target user.


## Example
The following example demonstrates out of order calls.


```c
#include <stdlib.h>
#include <sys/param.h>
#include <unistd.h>
#include <pwd.h>

void callSetuidAndCheck(int uid) {
    if (setuid(uid) != 0) {
        exit(1);
    }
}

void callSetgidAndCheck(int gid) {
    if (setgid(gid) != 0) {
        exit(1);
    }
}

/// Correct ways to drop priv.

void correctDropPrivInline() {
    if (setgroups(0, NULL)) {
        exit(1);
    }

    if (setgid(-2) != 0) {
        exit(1);
    }

    if (setuid(-2) != 0) {
        exit(1);
    }
}

void correctDropPrivInScope() {
    {
        if (setgroups(0, NULL)) {
            exit(1);
        }
    }

    {
        if (setgid(-2) != 0) {
            exit(1);
        }
    }

    {
        if (setuid(-2) != 0) {
            exit(1);
        }
    }
}

void correctOrderForInitgroups() {
    struct passwd *pw = getpwuid(0);
    if (pw) {
        if (initgroups(pw->pw_name, -2)) {
            exit(1);
        }
    } else {
        // Unhandled.
    }
    int rc = setuid(-2);
    if (rc) {
        exit(1);
    }
}

void correctDropPrivInScopeParent() {
    {
        callSetgidAndCheck(-2);
    }
    correctOrderForInitgroups();
}

void incorrectNoReturnCodeCheck() {
    int user = -2;
    if (user) {
        if (user) {
            int rc = setgid(user);
            (void)rc;
            initgroups("nobody", user);
        }
        if (user) {
            setuid(user);
        }
    }
}

void correctDropPrivInFunctionCall() {
    if (setgroups(0, NULL)) {
        exit(1);
    }

    callSetgidAndCheck(-2);
    callSetuidAndCheck(-2);
}

/// Incorrect, out of order gid and uid.
/// Calling uid before gid will fail.

void incorrectDropPrivOutOfOrderInline() {
    if (setuid(-2) != 0) {
        exit(1);
    }

    if (setgid(-2) != 0) {
        exit(1);
    }
}

void incorrectDropPrivOutOfOrderInScope() {
    {
        if (setuid(-2) != 0) {
            exit(1);
        }
    }

    setgid(-2);
}

void incorrectDropPrivOutOfOrderWithFunction() {
    callSetuidAndCheck(-2);

    if (setgid(-2) != 0) {
        exit(1);
    }
}

void incorrectDropPrivOutOfOrderWithFunction2() {
    callSetuidAndCheck(-2);
    callSetgidAndCheck(-2);
}

void incorrectDropPrivNoCheck() {
    setgid(-2);
    setuid(-2);
}

```

## References
* CERT C Coding Standard: [POS36-C. Observe correct revocation order while relinquishing privileges](https://wiki.sei.cmu.edu/confluence/display/c/POS36-C.+Observe+correct+revocation+order+while+relinquishing+privileges).
* CERT C Coding Standard: [POS37-C. Ensure that privilege relinquishment is successful](https://wiki.sei.cmu.edu/confluence/display/c/POS37-C.+Ensure+that+privilege+relinquishment+is+successful).
* Common Weakness Enumeration: [CWE-273](https://cwe.mitre.org/data/definitions/273.html).
