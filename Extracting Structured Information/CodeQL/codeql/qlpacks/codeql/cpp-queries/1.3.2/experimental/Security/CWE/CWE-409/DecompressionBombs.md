# User-controlled file decompression
Extracting Compressed files with any compression algorithm like gzip can cause denial of service attacks.

Attackers can compress a huge file consisting of repeated similiar bytes into a small compressed file.


## Recommendation
When you want to decompress a user-provided compressed file you must be careful about the decompression ratio or read these files within a loop byte by byte to be able to manage the decompressed size in each cycle of the loop.


## Example
Reading an uncompressed Gzip file within a loop and check for a threshold size in each cycle.


```cpp
#include "zlib.h"

void SafeGzread(gzFile inFileZ) {
    const int MAX_READ = 1024 * 1024 * 4;
    const int BUFFER_SIZE = 8192;
    unsigned char unzipBuffer[BUFFER_SIZE];
    unsigned int unzippedBytes;
    unsigned int totalRead = 0;
    while (true) {
        unzippedBytes = gzread(inFileZ, unzipBuffer, BUFFER_SIZE);
        totalRead += unzippedBytes;
        if (unzippedBytes <= 0) {
            break;
        }

        if (totalRead > MAX_READ) {
            // Possible decompression bomb, stop processing.
            break;
        } else {
            // process buffer
        }
    }
}

```
The following example is unsafe, as we do not check the uncompressed size.


```cpp
#include "zlib.h"

void UnsafeGzread(gzFile inFileZ) {
    const int BUFFER_SIZE = 8192;
    unsigned char unzipBuffer[BUFFER_SIZE];
    unsigned int unzippedBytes;
    while (true) {
        unzippedBytes = gzread(inFileZ, unzipBuffer, BUFFER_SIZE);
        if (unzippedBytes <= 0) {
            break;
        }

        // process buffer
    }
}

```

## References
* [Zlib documentation](https://zlib.net/manual.html)
* [An explanation of the attack](https://www.bamsoftware.com/hacks/zipbomb/)
* Common Weakness Enumeration: [CWE-409](https://cwe.mitre.org/data/definitions/409.html).
