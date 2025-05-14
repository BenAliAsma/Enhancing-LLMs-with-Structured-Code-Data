# Insecure randomness
Using a cryptographically weak pseudo-random number generator to generate a security-sensitive value, such as a password, makes it easier for an attacker to predict the value. Pseudo-random number generators generate a sequence of numbers that only approximates the properties of random numbers. The sequence is not truly random because it is completely determined by a relatively small set of initial values, the seed. If the random number generator is cryptographically weak, then this sequence may be easily predictable through outside observations.


## Recommendation
When generating values for use in security-sensitive contexts, it's essential to utilize a cryptographically secure pseudo-random number generator. As a general guideline, a value should be deemed "security-sensitive" if its predictability would empower an attacker to perform actions that would otherwise be beyond their reach. For instance, if an attacker could predict a newly generated user's random password, they would gain unauthorized access to that user's account. For Ruby, `SecureRandom` provides a cryptographically secure pseudo-random number generator. `rand` is not cryptographically secure, and should be avoided in security contexts. For contexts which are not security sensitive, `Random` may be preferable as it has a more convenient interface.


## Example
The following examples show different ways of generating a password.

The first example uses `Random.rand()` which is not for security purposes


```ruby
def generate_password()
  chars = ('a'..'z').to_a + ('A'..'Z').to_a + ('0'..'9').to_a + ['!', '@', '#', '$', '%']
  # BAD: rand is not cryptographically secure
  password = (1..10).collect { chars[rand(chars.size)] }.join
end

password = generate_password
```
In the second example, the password is generated using `SecureRandom.random_bytes()` which is a cryptographically secure method.


```ruby
require 'securerandom'

def generate_password()
  chars = ('a'..'z').to_a + ('A'..'Z').to_a + ('0'..'9').to_a + ['!', '@', '#', '$', '%']

  # GOOD: SecureRandom is cryptographically secure
  password = SecureRandom.random_bytes(10).each_byte.map do |byte|
    chars[byte % chars.length]
  end.join
end

password = generate_password()
```

## References
* Wikipedia: [Pseudo-random number generator](http://en.wikipedia.org/wiki/Pseudorandom_number_generator).
* Common Weakness Enumeration: [CWE-338](https://cwe.mitre.org/data/definitions/338.html).
* Ruby-doc: [Random](https://ruby-doc.org/core-3.1.2/Random.html).
* Common Weakness Enumeration: [CWE-338](https://cwe.mitre.org/data/definitions/338.html).
