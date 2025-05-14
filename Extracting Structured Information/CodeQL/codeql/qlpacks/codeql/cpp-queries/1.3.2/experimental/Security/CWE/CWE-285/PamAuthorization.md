# PAM Authorization bypass
Using only a call to `pam_authenticate` to check the validity of a login can lead to authorization bypass vulnerabilities.

A `pam_authenticate` only verifies the credentials of a user. It does not check if a user has an appropriate authorization to actually login. This means a user with a expired login or a password can still access the system.


## Recommendation
A call to `pam_authenticate` should be followed by a call to `pam_acct_mgmt` to check if a user is allowed to login.


## Example
In the following example, the code only checks the credentials of a user. Hence, in this case, a user expired with expired creds can still login. This can be verified by creating a new user account, expiring it with ``` chage -E0 `username`  ``` and then trying to log in.


```cpp
bool PamAuthGood(const std::string &username_in,
                 const std::string &password_in,
                 std::string &authenticated_username)
{

    struct pam_handle *pamh = nullptr; /* pam session handle */

    const char *username = username_in.c_str();
    int err = pam_start("test", username,
                        0, &pamh);
    if (err != PAM_SUCCESS)
    {
        return false;
    }

    err = pam_authenticate(pamh, 0); // BAD
    if (err != PAM_SUCCESS)
        return err;
    return true;
}

```
This can be avoided by calling `pam_acct_mgmt` call to verify access as has been done in the snippet shown below.


```cpp
bool PamAuthGood(const std::string &username_in,
                 const std::string &password_in,
                 std::string &authenticated_username)
{

    struct pam_handle *pamh = nullptr; /* pam session handle */

    const char *username = username_in.c_str();
    int err = pam_start("test", username,
                        0, &pamh);
    if (err != PAM_SUCCESS)
    {
        return false;
    }

    err = pam_authenticate(pamh, 0);
    if (err != PAM_SUCCESS)
        return err;

    err = pam_acct_mgmt(pamh, 0); // GOOD
    if (err != PAM_SUCCESS)
        return err;
    return true;
}

```

## References
* Man-Page: [pam_acct_mgmt](https://man7.org/linux/man-pages/man3/pam_acct_mgmt.3.html)
* Common Weakness Enumeration: [CWE-285](https://cwe.mitre.org/data/definitions/285.html).
