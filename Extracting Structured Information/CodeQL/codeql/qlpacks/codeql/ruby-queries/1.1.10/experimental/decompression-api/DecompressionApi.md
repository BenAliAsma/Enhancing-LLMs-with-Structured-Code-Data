# User-controlled file decompression
Decompression of user-controlled data without taking proper precaution can result in uncontrolled and massive decompression on the server, resulting in a denial of service.


## Recommendation
When decompressing files supplied by the user, make sure that you're checking the size of the incoming data chunks before writing to an output.


## Example
In this example, the size of the input buffer chunks and total size are checked before each chunk is written to the output.


```ruby
class UsersController < ActionController::Base
  def example_zlib_inflate
    MAX_ALLOWED_CHUNK_SIZE = 256
    MAX_ALLOWED_TOTAL_SIZE = 1024

    user_data = params[:data]
    output = []
    outsize = 0

    Zlib::Inflate.inflate(user_data) { |chunk|
      outsize += chunk.size
      if chunk.size < MAX_ALLOWED_CHUNK_SIZE && outsize < MAX_ALLOWED_TOTAL_SIZE
        output << chunk
      end
    }
  end
end
```

## References
* Common Weakness Enumeration: [CWE-409](https://cwe.mitre.org/data/definitions/409.html).
