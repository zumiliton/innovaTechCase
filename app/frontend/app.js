const imageInput = document.getElementById("imageInput");
const preview = document.getElementById("preview");
const previewContainer = document.getElementById("previewContainer");
const uploadButton = document.getElementById("uploadButton");
const result = document.getElementById("result");

let selectedFile = null;


imageInput.addEventListener("change", () => {

    const file = imageInput.files[0];

    if (!file) {
        return;
    }

    selectedFile = file;

    preview.src = URL.createObjectURL(file);

    previewContainer.classList.remove("hidden");
    uploadButton.disabled = false;

    result.classList.add("hidden");
});


uploadButton.addEventListener("click", async () => {

    if (!selectedFile) {
        return;
    }

    uploadButton.disabled = true;
    uploadButton.textContent = "Processing...";

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {

        const response = await fetch("/api/upload", {
            method: "POST",
            body: formData,
        });

        const data = await response.json();

        result.textContent =
            `${data.filename} received successfully`;

        result.classList.remove("hidden");

    } catch (error) {

        result.textContent =
            "Something went wrong while uploading the image.";

        result.classList.remove("hidden");

    } finally {

        uploadButton.disabled = false;
        uploadButton.textContent = "Identify board";
    }
});