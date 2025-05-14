# Uncontrolled file decompression
Extracting Compressed files with any compression algorithm like gzip can cause a denial of service attack.

Attackers can create a huge file by just repeating a single byte and compress it to a small file.


## Recommendation
When decompressing a user-provided compressed file, verify the decompression ratio or decompress the files within a loop byte by byte to be able to manage the decompressed size in each cycle of the loop.


## Example
In the following example, the decompressed file size is not checked before decompression, exposing the application to a denial of service.


```java
package org.example;

import java.nio.file.StandardCopyOption;
import java.util.Enumeration;
import java.io.IOException;
import java.util.zip.*;
import java.util.zip.ZipEntry;
import java.io.File;
import java.nio.file.Files;


class BadExample {
    public static void ZipInputStreamUnSafe(String filename) throws IOException {
        File f = new File(filename);
        try (ZipFile zipFile = new ZipFile(f)) {
            Enumeration<? extends ZipEntry> entries = zipFile.entries();

            while (entries.hasMoreElements()) {
                ZipEntry ze = entries.nextElement();
                File out = new File("./tmp/tmp.txt");
                Files.copy(zipFile.getInputStream(ze), out.toPath(), StandardCopyOption.REPLACE_EXISTING);
            }
        }
    }
}
```
A better approach is shown in the following example, where a ZIP file is read within a loop and a size threshold is checked every cycle.


```java
import java.util.zip.*;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.util.zip.ZipEntry;

public class GoodExample {
    public static void ZipInputStreamSafe(String filename) throws IOException {
        int UncompressedSizeThreshold = 10 * 1024 * 1024; // 10MB
        int BUFFERSIZE = 256;
        FileInputStream fis = new FileInputStream(filename);
        try (ZipInputStream zis = new ZipInputStream(new BufferedInputStream(fis))) {
            ZipEntry entry;
            while ((entry = zis.getNextEntry()) != null) {
                int count;
                byte[] data = new byte[BUFFERSIZE];
                FileOutputStream fos = new FileOutputStream(entry.getName());
                BufferedOutputStream dest = new BufferedOutputStream(fos, BUFFERSIZE);
                int totalRead = 0;
                while ((count = zis.read(data, 0, BUFFERSIZE)) != -1) {
                    totalRead = totalRead + count;
                    if (totalRead > UncompressedSizeThreshold) {
                        System.out.println("This Compressed file can be a bomb!");
                        break;
                    }
                    dest.write(data, 0, count);
                }
                dest.flush();
                dest.close();
                zis.closeEntry();
            }
        }
    }
}
```

## References
* [CVE-2022-4565](https://github.com/advisories/GHSA-47vx-fqr5-j2gw)
* David Fifield: [A better zip bomb](https://www.bamsoftware.com/hacks/zipbomb/).
* Common Weakness Enumeration: [CWE-409](https://cwe.mitre.org/data/definitions/409.html).
