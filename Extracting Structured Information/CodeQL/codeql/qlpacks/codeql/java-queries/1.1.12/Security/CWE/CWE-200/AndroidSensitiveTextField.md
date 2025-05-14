# Exposure of sensitive information to UI text views
Sensitive information such as passwords should not be displayed in UI components unless explicitly required, to mitigate shoulder-surfing attacks.


## Recommendation
For editable text fields containing sensitive information, the `inputType` should be set to `textPassword` or similar to ensure it is properly masked. Otherwise, sensitive data that must be displayed should be hidden by default, and only revealed based on an explicit user action.


## Example
In the following (bad) case, sensitive information in `password` is exposed to the `TextView`.


```java
TextView pwView = getViewById(R.id.pw_text);
pwView.setText("Your password is: " + password);
```
In the following (good) case, the user must press a button to reveal sensitive information.


```java
TextView pwView = findViewById(R.id.pw_text);
pwView.setVisibility(View.INVISIBLE);
pwView.setText("Your password is: " + password);

Button showButton = findViewById(R.id.show_pw_button);
showButton.setOnClickListener(new View.OnClickListener() {
    public void onClick(View v) {
      pwView.setVisibility(View.VISIBLE);
    }
});

```

## References
* OWASP Mobile Application Security: [Android Data Storage - UI Components](https://mas.owasp.org/MASTG/Android/0x05d-Testing-Data-Storage/#ui-components)
* Common Weakness Enumeration: [CWE-200](https://cwe.mitre.org/data/definitions/200.html).
