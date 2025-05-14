# Insecure Mass Assignment
Operations that allow for mass assignment (setting multiple attributes of an object using a hash), such as `ActiveRecord::Base.new`, should take care not to allow arbitrary parameters to be set by the user. Otherwise, unintended attributes may be set, such as an `is_admin` field for a `User` object.


## Recommendation
When using a mass assignment operation from user supplied parameters, use `ActionController::Parameters#permit` to restrict the possible parameters a user can supply, rather than `ActionController::Parameters#permit!`, which permits arbitrary parameters to be used for mass assignment.


## Example
In the following example, `permit!` is used which allows arbitrary parameters to be supplied by the user.


```ruby
class UserController < ActionController::Base
    def create
        # BAD: arbitrary params are permitted to be used for this assignment
        User.new(user_params).save!
    end

    def user_params
        params.require(:user).permit!
    end
end
```


In the following example, only specific parameters are permitted, so the mass assignment is safe.


```ruby
class UserController < ActionController::Base
    def create
        # GOOD: the permitted parameters are explicitly specified
        User.new(user_params).save!
    end

    def user_params
        params.require(:user).permit(:name, :email)
    end
end
```

## References
* Rails guides: [Strong Parameters](https://guides.rubyonrails.org/action_controller_overview.html#strong-parameters).
* Common Weakness Enumeration: [CWE-915](https://cwe.mitre.org/data/definitions/915.html).
