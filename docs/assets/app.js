// Add a copy button to every code block. No dependencies.
document.querySelectorAll("pre").forEach(function (pre) {
    if (!navigator.clipboard) return;
    var button = document.createElement("button");
    button.type = "button";
    button.className = "copy";
    button.textContent = "Copy";
    button.setAttribute("aria-label", "Copy code to clipboard");

    var reset = function () {
        button.textContent = "Copy";
    };

    button.addEventListener("click", function () {
        var code = pre.querySelector("code");
        var text = code ? code.innerText : pre.innerText;
        navigator.clipboard.writeText(text).then(
            function () {
                button.textContent = "Copied";
                setTimeout(reset, 1500);
            },
            function () {
                button.textContent = "Failed";
                setTimeout(reset, 1500);
            }
        );
    });

    pre.appendChild(button);
});
