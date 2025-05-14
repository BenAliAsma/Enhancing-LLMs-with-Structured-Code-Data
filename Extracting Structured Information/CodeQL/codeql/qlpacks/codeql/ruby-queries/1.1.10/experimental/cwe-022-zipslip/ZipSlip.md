# Arbitrary file access during archive extraction ("Zip Slip")
Extracting files from a malicious zip file, or similar type of archive, is at risk of directory traversal attacks if filenames from the archive are not properly validated.

Tar archives contain archive entries representing each file in the archive. These entries include a file path for the entry, but these file paths are not restricted and may contain unexpected special elements such as the directory traversal element (`..`). If these file paths are used to create a filesystem path, then a file operation may happen in an unexpected location. This can result in sensitive information being revealed or deleted, or an attacker being able to influence behavior by modifying unexpected files.

For example, if a tar archive contains a file entry `..\sneaky-file`, and the tar archive is extracted to the directory `c:\output`, then naively combining the paths would result in an output file path of `c:\output\..\sneaky-file`, which would cause the file to be written to `c:\sneaky-file`.


## Recommendation
Ensure that output paths constructed from tar archive entries are validated to prevent writing files to unexpected locations.

The recommended way of writing an output file from a tar archive entry is to check that `".."` does not occur in the path.


## Example
In this example an archive is extracted without validating file paths. If `archive.tar` contained relative paths (for instance, if it were created by something like `tar -cf archive.tar ../file.txt`) then executing this code could write to locations outside the destination directory.


```ruby
class FilesController < ActionController::Base
  def zipFileUnsafe
    path = params[:path]
    Zip::File.open(path).each do |entry|
      File.open(entry.name, "wb") do |os|
        entry.read
      end
    end
  end

  def tarReaderUnsafe
    path = params[:path]
    file_stream = IO.new(IO.sysopen(path))
    tarfile = Gem::Package::TarReader.new(file_stream)
    tarfile.each do |entry|
      ::File.open(entry.full_name, "wb") do |os|
        entry.read
      end
    end
  end  
end

```
To fix this vulnerability, we need to check that the path does not contain any `".."` elements in it.


```ruby
class FilesController < ActionController::Base
  def zipFileSafe
    path = params[:path]
    Zip::File.open(path).each do |entry|
      entry_path = entry.name
      next if !File.expand_path(entry_path).start_with?('/safepath/')
      File.open(entry_path, "wb") do |os|
        entry.read
      end
    end
  end

  def tarReaderSafe
    path = params[:path]
    file_stream = IO.new(IO.sysopen(path))
    tarfile = Gem::Package::TarReader.new(file_stream)
    tarfile.each do |entry|
      entry_path = entry.full_name
      raise ExtractFailed if entry_path != "/safepath"
      ::File.open(entry_path, "wb") do |os|
        entry.read
      end
    end
  end  
end

```

## References
* Snyk: [Zip Slip Vulnerability](https://snyk.io/research/zip-slip-vulnerability).
* OWASP: [Path Traversal](https://owasp.org/www-community/attacks/Path_Traversal).
* class [Gem::Package::TarReader](https://docs.ruby-lang.org/en/2.4.0/Gem/Package/TarReader.html).
* class [Zlib::GzipReader](https://ruby-doc.org/stdlib-2.4.0/libdoc/zlib/rdoc/Zlib/GzipReader.html).
* class [Zip::File](https://www.rubydoc.info/github/rubyzip/rubyzip/Zip/File).
* Common Weakness Enumeration: [CWE-22](https://cwe.mitre.org/data/definitions/22.html).
