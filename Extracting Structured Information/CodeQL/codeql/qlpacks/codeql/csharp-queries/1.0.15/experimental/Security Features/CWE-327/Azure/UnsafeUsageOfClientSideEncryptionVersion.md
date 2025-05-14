# Unsafe usage of v1 version of Azure Storage client-side encryption (CVE-2022-30187).
Azure Storage .NET, Java, and Python SDKs support encryption on the client with a customer-managed key that is maintained in Azure Key Vault or another key store.

Current release versions of the Azure Storage SDKs use cipher block chaining (CBC mode) for client-side encryption (referred to as `v1`).


## Recommendation
Consider switching to `v2` client-side encryption.


## Example

```csharp

{
    SymmetricKey aesKey = new SymmetricKey(kid: "symencryptionkey");

    // BAD: Using the outdated client side encryption version V1_0
    BlobEncryptionPolicy uploadPolicy = new BlobEncryptionPolicy(key: aesKey, keyResolver: null);
    BlobRequestOptions uploadOptions = new BlobRequestOptions() { EncryptionPolicy = uploadPolicy };

    MemoryStream stream = new MemoryStream(buffer);
    blob.UploadFromStream(stream, length: size, accessCondition: null, options: uploadOptions);
}

var client = new BlobClient(myConnectionString, new SpecializedBlobClientOptions()
{
    // BAD: Using an outdated SDK that does not support client side encryption version V2_0
    ClientSideEncryption = new ClientSideEncryptionOptions() 
    {
        KeyEncryptionKey = myKey,
        KeyResolver = myKeyResolver,
        KeyWrapAlgorithm = myKeyWrapAlgorithm
    }
});

var client = new BlobClient(myConnectionString, new SpecializedBlobClientOptions()
{
    // BAD: Using the outdated client side encryption version V1_0
    ClientSideEncryption = new ClientSideEncryptionOptions(ClientSideEncryptionVersion.V1_0) 
    {
        KeyEncryptionKey = myKey,
        KeyResolver = myKeyResolver,
        KeyWrapAlgorithm = myKeyWrapAlgorithm
    }
});

var client = new BlobClient(myConnectionString, new SpecializedBlobClientOptions()
{
    // GOOD: Using client side encryption version V2_0
    ClientSideEncryption = new ClientSideEncryptionOptions(ClientSideEncryptionVersion.V2_0) 
    {
        KeyEncryptionKey = myKey,
        KeyResolver = myKeyResolver,
        KeyWrapAlgorithm = myKeyWrapAlgorithm
    }
});
```

## References
* [Azure Storage Client Encryption Blog.](http://aka.ms/azstorageclientencryptionblog)
* [CVE-2022-30187](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2022-30187)
* Common Weakness Enumeration: [CWE-327](https://cwe.mitre.org/data/definitions/327.html).
