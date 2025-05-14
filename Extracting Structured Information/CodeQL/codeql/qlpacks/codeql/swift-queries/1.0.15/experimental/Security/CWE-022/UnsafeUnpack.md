# Arbitrary file write during a zip extraction from a user controlled source
Unpacking files from a malicious zip without properly validating that the destination file path is within the destination directory, or allowing symlinks to point to files outside the extraction directory, allows an attacker to extract files to arbitrary locations outside the extraction directory. This helps overwrite sensitive user data and, in some cases, can lead to code execution if an attacker overwrites an application's shared object file.


## Recommendation
Consider using a safer module, such as: `ZIPArchive`


## Example
The following examples unpacks a remote zip using \`Zip.unzipFile()\` which is vulnerable to path traversal.


```swift
import Foundation
import Zip


func unzipFile(at sourcePath: String, to destinationPath: String) {
    do {
        let remoteURL = URL(string: "https://example.com/")!

        let source  = URL(fileURLWithPath: sourcePath)
        let destination = URL(fileURLWithPath: destinationPath)

        // Malicious zip is downloaded 
        try Data(contentsOf: remoteURL).write(to: source)

        let fileManager = FileManager()
        // Malicious zip is unpacked
        try Zip.unzipFile(source, destination: destination, overwrite: true, password: nil)
        } catch {
    }
}

func main() {
    let sourcePath = "/sourcePath" 
    let destinationPath = "/destinationPath" 
    unzipFile(at: sourcePath, to: destinationPath)
}

main()
```
The following examples unpacks a remote zip using \`fileManager.unzipItem()\` which is vulnerable to symlink path traversal.


```swift
import Foundation
import ZIPFoundation


func unzipFile(at sourcePath: String, to destinationPath: String) {
    do {
        let remoteURL = URL(string: "https://example.com/")!

        let source  = URL(fileURLWithPath: sourcePath)
        let destination = URL(fileURLWithPath: destinationPath)

        // Malicious zip is downloaded 
        try Data(contentsOf: remoteURL).write(to: source)

        let fileManager = FileManager()
        // Malicious zip is unpacked
        try fileManager.unzipItem(at:source, to: destination)
        } catch {
    }
}

func main() {
    let sourcePath = "/sourcePath" 
    let destinationPath = "/destinationPath" 
    unzipFile(at: sourcePath, to: destinationPath)
}

main()
```
Consider using a safer module, such as: `ZIPArchive`


```swift
import Foundation
import ZipArchive

func unzipFile(at sourcePath: String, to destinationPath: String) {
    do {
        let remoteURL = URL(string: "https://example.com/")!

        let source  = URL(fileURLWithPath: sourcePath)

        // Malicious zip is downloaded 
        try Data(contentsOf: remoteURL).write(to: source)

        // ZipArchive is safe
        try SSZipArchive.unzipFile(atPath: sourcePath, toDestination: destinationPath, delegate: self)
        } catch {
    }
}

func main() {
    let sourcePath = "/sourcePath" 
    let destinationPath = "/destinationPath" 
    unzipFile(at: sourcePath, to: destinationPath)
}

main()
```

## References
* Ostorlab: [Zip Packages Exploitation](https://blog.ostorlab.co/zip-packages-exploitation.html).
* Common Weakness Enumeration: [CWE-22](https://cwe.mitre.org/data/definitions/22.html).
