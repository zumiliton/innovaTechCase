const messages = [];

let lastUserMessage = "";
let botMessage = "";

let talking = true;
let selectedFile = null;

// ==================================================
// DOM ELEMENTS
// ==================================================

const imageInput =
document.getElementById("imageInput");

const preview =
document.getElementById("preview");

const previewContainer =
document.getElementById("previewContainer");

const uploadButton =
document.getElementById("uploadButton");

const result =
document.getElementById("result");

const chatMessages =
document.getElementById("chatMessages");

const emptyChat =
document.getElementById("emptyChat");

const questionInput =
document.getElementById("questionInput");

const askButton =
document.getElementById("askButton");

const chatStatus =
document.getElementById("chatStatus");

// ==================================================
// IMAGE SELECTION
// ==================================================

imageInput.addEventListener("change", () => {


const file =
    imageInput.files[0];

if (!file) {
    return;
}

selectedFile = file;

preview.src =
    URL.createObjectURL(file);

previewContainer.classList.remove(
    "hidden"
);

uploadButton.disabled = false;

result.classList.add(
    "hidden"
);

resetChat();


});

// ==================================================
// BOARD IDENTIFICATION
// ==================================================

uploadButton.addEventListener(
"click",
identifyBoard
);

async function identifyBoard() {


if (!selectedFile) {
    return;
}

uploadButton.disabled = true;

uploadButton.textContent =
    "Identifying...";


const formData =
    new FormData();

formData.append(
    "file",
    selectedFile
);


try {

    const response =
        await fetch(
            "/api/predict",
            {
                method: "POST",
                body: formData
            }
        );


    const data =
        await response.json();


    if (!response.ok) {

        throw new Error(
            data.detail ||
            data.error ||
            "Prediction failed"
        );

    }


    // ------------------------------------------
    // Prediction result
    // ------------------------------------------

    const confidence =
        (
            data.confidence * 100
        ).toFixed(1);


    result.innerHTML = `
        <strong>
            Arduino ${data.board}
        </strong>

        <span>
            Confidence: ${confidence}%
        </span>
    `;


    result.classList.remove(
        "hidden"
    );


    // ------------------------------------------
    // Enable chatbot
    // ------------------------------------------

    chatStatus.innerHTML = `
        <span class="status-dot active"></span>
        Arduino ${data.board} identified
    `;


    questionInput.disabled = false;

    askButton.disabled = false;


    questionInput.placeholder =
        `Ask something about Arduino ${data.board}...`;


    questionInput.focus();

}
catch (error) {

    console.error(
        "[VISION]",
        error
    );


    result.innerHTML = `
        <strong>
            Identification failed
        </strong>

        <span>
            ${error.message}
        </span>
    `;


    result.classList.remove(
        "hidden"
    );

}
finally {

    uploadButton.disabled = false;

    uploadButton.textContent =
        "Identify board";

}


}

// ==================================================
// CHATBOT
// ==================================================

// This replaces the original chatbotResponse()
// from the CodePen template.
//
// Instead of generating a hardcoded response,
// we call your FastAPI backend.

async function chatbotResponse() {


botMessage = "";


try {

    const response =
        await fetch(
            "/api/ask",
            {
                method: "POST",

                headers: {
                    "Content-Type":
                        "application/json"
                },

                body: JSON.stringify({
                    question:
                        lastUserMessage
                })
            }
        );


    const data =
        await response.json();


    if (!response.ok) {

        throw new Error(
            data.detail ||
            data.error ||
            "Assistant request failed"
        );

    }


    if (data.error) {

        throw new Error(
            data.error
        );

    }


    botMessage =
        data.answer;


    return data;

}
catch (error) {

    console.error(
        "[CHAT]",
        error
    );


    botMessage =
        `Error: ${error.message}`;


    return {
        answer: botMessage,
        sources: []
    };

}


}

// ==================================================
// NEW CHAT ENTRY
// ==================================================

async function newEntry() {


const input =
    questionInput.value.trim();


if (input === "") {
    return;
}


// ------------------------------------------
// Save user message
// ------------------------------------------

lastUserMessage =
    input;


questionInput.value = "";


messages.push(
    lastUserMessage
);


// ------------------------------------------
// Display user message
// ------------------------------------------

addMessage(
    "user",
    lastUserMessage
);


// ------------------------------------------
// Disable input while thinking
// ------------------------------------------

questionInput.disabled = true;

askButton.disabled = true;


// ------------------------------------------
// Thinking message
// ------------------------------------------

const thinking =
    addMessage(
        "assistant",
        "Thinking..."
    );


// ------------------------------------------
// Ask backend
// ------------------------------------------

const data =
    await chatbotResponse();


// ------------------------------------------
// Remove thinking
// ------------------------------------------

thinking.remove();


// ------------------------------------------
// Save bot response
// ------------------------------------------

messages.push(
    botMessage
);


// ------------------------------------------
// Display response
// ------------------------------------------

addMessage(
    "assistant",
    botMessage,
    data.sources
);


// ------------------------------------------
// Text-to-speech
// ------------------------------------------

Speech(
    botMessage
);


// ------------------------------------------
// Re-enable input
// ------------------------------------------

questionInput.disabled = false;

askButton.disabled = false;

questionInput.focus();


}

// ==================================================
// ADD MESSAGE
// ==================================================

function addMessage(
role,
content,
sources = []
) {


if (emptyChat) {

    emptyChat.remove();

}


const message =
    document.createElement(
        "div"
    );


message.classList.add(
    "chat-message",
    role
);


// ------------------------------------------
// Label
// ------------------------------------------

const label =
    document.createElement(
        "div"
    );


label.classList.add(
    "message-label"
);


label.textContent =
    role === "user"
        ? "You"
        : "Arduino Assistant";


// ------------------------------------------
// Bubble
// ------------------------------------------

const bubble =
    document.createElement(
        "div"
    );


bubble.classList.add(
    "message-bubble"
);


bubble.textContent =
    content;


message.appendChild(
    label
);

message.appendChild(
    bubble
);


// ------------------------------------------
// Sources
// ------------------------------------------

if (
    role === "assistant" &&
    sources &&
    sources.length > 0
) {

    const sourcesContainer =
        document.createElement(
            "div"
        );


    sourcesContainer.classList.add(
        "message-sources"
    );


    const title =
        document.createElement(
            "strong"
        );


    title.textContent =
        "Sources";


    sourcesContainer.appendChild(
        title
    );


    sources.forEach(
        (source) => {

            const item =
                document.createElement(
                    "div"
                );


            item.classList.add(
                "source-item"
            );


            item.textContent =
                `${source.source} · ${source.section}`;


            sourcesContainer.appendChild(
                item
            );

        }
    );


    message.appendChild(
        sourcesContainer
    );

}


chatMessages.appendChild(
    message
);


// ------------------------------------------
// Scroll to bottom
// ------------------------------------------

chatMessages.scrollTop =
    chatMessages.scrollHeight;


return message;


}

// ==================================================
// ASK BUTTON
// ==================================================

askButton.addEventListener(
"click",
newEntry
);

// ==================================================
// ENTER KEY
// ==================================================

questionInput.addEventListener(
"keydown",
(event) => {


    if (
        event.key === "Enter" &&
        !event.shiftKey
    ) {

        event.preventDefault();

        newEntry();

    }

}


);

// ==================================================
// TEXT TO SPEECH
// ==================================================

function Speech(say) {


if (
    "speechSynthesis" in window &&
    talking
) {

    const utterance =
        new SpeechSynthesisUtterance(
            say
        );


    utterance.lang =
        "en-US";


    speechSynthesis.speak(
        utterance
    );

}


}

// ==================================================
// RESET CHAT
// ==================================================

function resetChat() {


messages.length = 0;

lastUserMessage = "";

botMessage = "";


chatMessages.innerHTML = `
    <div
        id="emptyChat"
        class="empty-chat"
    >

        <div class="empty-chat-icon">
            💬
        </div>

        <strong>
            Ask about your Arduino
        </strong>

        <p>
            Identify a board first, then ask
            questions about its specifications
            and documentation.
        </p>

    </div>
`;


chatStatus.innerHTML = `
    <span class="status-dot"></span>
    Identify a board to start chatting
`;


questionInput.disabled = true;

askButton.disabled = true;


questionInput.placeholder =
    "Identify a board first...";


}

// ==================================================
// INITIAL STATE
// ==================================================

questionInput.disabled = true;

askButton.disabled = true;
