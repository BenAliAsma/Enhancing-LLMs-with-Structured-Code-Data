# Unsafe usage of v1 version of Azure Storage client-side encryption (CVE-2022-30187).
Azure Storage .NET, Java, and Python SDKs support encryption on the client with a customer-managed key that is maintained in Azure Key Vault or another key store.

The Azure Storage SDK version 12.18.0 or later supports version `V2` for client-side encryption. All previous versions of Azure Storage SDK only support client-side encryption `V1` which is unsafe.


## Recommendation
Consider switching to `V2` client-side encryption.


## Example

```java

// BAD: Using an outdated SDK that does not support client side encryption version V2_0
new EncryptedBlobClientBuilder()
        .blobClient(blobClient)
        .key(resolver.buildAsyncKeyEncryptionKey(keyid).block(), keyWrapAlgorithm)
        .buildEncryptedBlobClient()
        .uploadWithResponse(new BlobParallelUploadOptions(data)
                        .setMetadata(metadata)
                        .setHeaders(headers)
                        .setTags(tags)
                        .setTier(tier)
                        .setRequestConditions(requestConditions)
                        .setComputeMd5(computeMd5)
                        .setParallelTransferOptions(parallelTransferOptions),
                timeout, context);

// BAD: Using the deprecatedd client side encryption version V1_0
new EncryptedBlobClientBuilder(EncryptionVersion.V1)
        .blobClient(blobClient)
        .key(resolver.buildAsyncKeyEncryptionKey(keyid).block(), keyWrapAlgorithm)
        .buildEncryptedBlobClient()
        .uploadWithResponse(new BlobParallelUploadOptions(data)
                        .setMetadata(metadata)
                        .setHeaders(headers)
                        .setTags(tags)
                        .setTier(tier)
                        .setRequestConditions(requestConditions)
                        .setComputeMd5(computeMd5)
                        .setParallelTransferOptions(parallelTransferOptions),
                timeout, context);


// GOOD: Using client side encryption version V2_0
new EncryptedBlobClientBuilder(EncryptionVersion.V2)
        .blobClient(blobClient)
        .key(resolver.buildAsyncKeyEncryptionKey(keyid).block(), keyWrapAlgorithm)
        .buildEncryptedBlobClient()
        .uploadWithResponse(new BlobParallelUploadOptions(data)
                        .setMetadata(metadata)
                        .setHeaders(headers)
                        .setTags(tags)
                        .setTier(tier)
                        .setRequestConditions(requestConditions)
                        .setComputeMd5(computeMd5)
                        .setParallelTransferOptions(parallelTransferOptions),
                timeout, context);

```

## References
* [Azure Storage Client Encryption Blog.](http://aka.ms/azstorageclientencryptionblog)
* [CVE-2022-30187](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2022-30187)
* Common Weakness Enumeration: [CWE-327](https://cwe.mitre.org/data/definitions/327.html).
