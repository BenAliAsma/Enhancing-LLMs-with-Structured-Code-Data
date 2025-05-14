# User-controlled file decompression
Extracting Compressed files with any compression algorithm like gzip can cause to denial of service attacks.

Attackers can compress a huge file which created by repeated similiar byte and convert it to a small compressed file.


## Recommendation
When you want to decompress a user-provided compressed file you must be careful about the decompression ratio or read these files within a loop byte by byte to be able to manage the decompressed size in each cycle of the loop.

Please read official RubyZip Documentation [here](https://github.com/rubyzip/rubyzip/#size-validation)


## Example
Rubyzip: According to [official](https://github.com/rubyzip/rubyzip/#reading-a-zip-file) Documentation


```ruby
MAX_FILE_SIZE = 10 * 1024**2 # 10MiB
MAX_FILES = 100
Zip::File.open('foo.zip') do |zip_file|
  num_files = 0
  zip_file.each do |entry|
    num_files += 1 if entry.file?
    raise 'Too many extracted files' if num_files > MAX_FILES
    raise 'File too large when extracted' if entry.size > MAX_FILE_SIZE
    entry.extract
  end
end
```

```ruby
# "Note that if you use the lower level Zip::InputStream interface, rubyzip does not check the entry sizes"
zip_stream = Zip::InputStream.new(File.open('file.zip'))
while entry = zip_stream.get_next_entry
  # All required operations on `entry` go here.
end
```

## References
* [CVE-2023-22898](https://www.cvedetails.com/cve/CVE-2022-3759/) [Gitlab issue](https://gitlab.com/gitlab-org/gitlab/-/issues/379633)
* [A great research to gain more impact by this kind of attack](https://www.bamsoftware.com/hacks/zipbomb/)
* Common Weakness Enumeration: [CWE-409](https://cwe.mitre.org/data/definitions/409.html).
