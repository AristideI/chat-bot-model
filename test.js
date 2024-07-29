const url = "http://127.0.0.1:8000/chat";
const message = { message: "hello" };

fetch(url, {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
  },
  body: JSON.stringify(message),
})
  .then((response) => response.json())
  .then((data) => console.log(data));
