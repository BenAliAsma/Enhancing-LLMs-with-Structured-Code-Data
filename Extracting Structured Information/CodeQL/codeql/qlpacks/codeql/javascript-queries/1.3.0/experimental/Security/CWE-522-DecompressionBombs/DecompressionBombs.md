# User-controlled file decompression
Extracting Compressed files with any compression algorithm like gzip can cause to denial of service attacks.

Attackers can compress a huge file which created by repeated similiar byte and convert it to a small compressed file.


## Recommendation
When you want to decompress a user-provided compressed file you must be careful about the decompression ratio or read these files within a loop byte by byte to be able to manage the decompressed size in each cycle of the loop.


## Example
JsZip: check uncompressedSize Object Field before extraction.


```javascript
const jszipp = require("jszip");
function zipBombSafe(zipFile) {
    jszipp.loadAsync(zipFile.data).then(function (zip) {
        if (zip.file("10GB")["_data"]["uncompressedSize"] > 1024 * 1024 * 8) {
            console.log("error")
        }
        zip.file("10GB").async("uint8array").then(function (u8) {
            console.log(u8);
        });
    });
}
```
nodejs Zlib: use [maxOutputLength option](https://nodejs.org/dist/latest-v18.x/docs/api/zlib.html#class-options) which it'll limit the buffer read size


```javascript
const zlib = require("zlib");

zlib.gunzip(
    inputZipFile.data,
    { maxOutputLength: 1024 * 1024 * 5 },
    (err, buffer) => {
        doSomeThingWithData(buffer);
    });
zlib.gunzipSync(inputZipFile.data, { maxOutputLength: 1024 * 1024 * 5 });

inputZipFile.pipe(zlib.createGunzip({ maxOutputLength: 1024 * 1024 * 5 })).pipe(outputFile);
```
node-tar: use [maxReadSize option](https://github.com/isaacs/node-tar/blob/8c5af15e43a769fd24aa7f1c84d93e54824d19d2/lib/list.js#L90) which it'll limit the buffer read size


```javascript
const tar = require("tar");

tar.x({
    file: tarFileName,
    strip: 1,
    C: 'some-dir',
    maxReadSize: 16 * 1024 * 1024 // 16 MB
})
```

## References
* [CVE-2017-16129](https://github.com/advisories/GHSA-8225-6cvr-8pqp)
* [A great research to gain more impact by this kind of attacks](https://www.bamsoftware.com/hacks/zipbomb/)
* Common Weakness Enumeration: [CWE-522](https://cwe.mitre.org/data/definitions/522.html).
