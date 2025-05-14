# Cleartext storage of sensitive information in file
Sensitive information that is stored unencrypted is accessible to an attacker who gains access to the storage.


## Recommendation
Ensure that sensitive information is always encrypted before being stored to a file or transmitted over the network. It may be wise to encrypt information before it is put into a buffer that may be readable in memory.

In general, decrypt sensitive information only at the point where it is necessary for it to be used in cleartext.


## Example
The following example shows two ways of storing user credentials in a file. In the 'BAD' case, the credentials are simply stored in cleartext. In the 'GOOD' case, the credentials are encrypted before storing them.


```c
#include <sodium.h>
#include <stdio.h>
#include <string.h>

void writeCredentialsBad(FILE *file, const char *cleartextCredentials) {
  // BAD: write password to disk in cleartext
  fputs(cleartextCredentials, file);
}

int writeCredentialsGood(FILE *file, const char *cleartextCredentials, const unsigned char *key, const unsigned char *nonce) {
  size_t credentialsLen = strlen(cleartextCredentials);
  size_t ciphertext_len = crypto_secretbox_MACBYTES + credentialsLen;
  unsigned char *ciphertext = malloc(ciphertext_len);
  if (!ciphertext) {
    logError();
    return -1;
  }

  // encrypt the password first
  if (crypto_secretbox_easy(ciphertext, (const unsigned char *)cleartextCredentials, credentialsLen, nonce, key) != 0) {
    free(ciphertext);
    logError();
    return -1;
  }

  // GOOD: write encrypted password to disk
  fwrite(ciphertext, 1, ciphertext_len, file);

  free(ciphertext);
  return 0;
}

```
Note that for the 'GOOD' example to work we need to link against an encryption library (in this case libsodium), initialize it with a call to `sodium_init`, and create the key and nonce with `crypto_secretbox_keygen` and `randombytes_buf` respectively. We also need to store those details securely so they can be used for decryption.


## References
* M. Dowd, J. McDonald and J. Schuhm, *The Art of Software Security Assessment*, 1st Edition, Chapter 2 - 'Common Vulnerabilities of Encryption', p. 43. Addison Wesley, 2006.
* M. Howard and D. LeBlanc, *Writing Secure Code*, 2nd Edition, Chapter 9 - 'Protecting Secret Data', p. 299. Microsoft, 2002.
* Common Weakness Enumeration: [CWE-260](https://cwe.mitre.org/data/definitions/260.html).
* Common Weakness Enumeration: [CWE-313](https://cwe.mitre.org/data/definitions/313.html).
