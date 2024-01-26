var WorkflowContainer = {
    createCodeHeader: function(text) {
        const div = document.createElement("div");
        div.className = "codeHeader";
        div.innerHTML =
            '<span class="language">'+ text +'</span>'+
            '<a class="btn border-0 code-action" href="#" title="Copy">'+
            '  <i class="bi bi-clipboard"></i>'+
            '</a>';
        return div;
    },
    setCopyAlert: function(element) {
        const copyIcon = element.querySelector("i");
        element.classList.add("link-success");
        copyIcon.classList.add("bi-check-lg");
        copyIcon.classList.remove("bi-clipboard");
        setTimeout(function() {
            copyIcon.classList.remove("bi-check-lg");
            copyIcon.classList.add("bi-clipboard");
            element.classList.remove("link-success");
        }, 1000);
    },
    renderElement: function(element) {
        const img = element.querySelector("img");
        const workflowPath = img.src;
        img.src = workflowPath.replace(/\.[^.]+$/, ".svg");

        const codeHeader = WorkflowContainer.createCodeHeader("Workflow");
        const button = codeHeader.querySelector("a");
        button.addEventListener("click", (e) => {
            e.preventDefault();
            fetch(workflowPath).then(req => req.text()).then(contents => {
                navigator.clipboard.writeText(contents);
                WorkflowContainer.setCopyAlert(button);
            });
        });

        const wrap = document.createElement("p");
        const parent = element.parentElement
        parent.insertBefore(wrap, element);
        wrap.appendChild(codeHeader);
        wrap.appendChild(element);
    },
    init: async function() {
        for (const element of document.getElementsByClassName("workflow")) {
            WorkflowContainer.renderElement(element)
        }
    }
}

export default {
    defaultTheme: 'auto',
    iconLinks: [{
        icon: 'github',
        href: 'https://github.com/bonsai-rx/machinelearning',
        title: 'GitHub'
    }]
}
WorkflowContainer.init();