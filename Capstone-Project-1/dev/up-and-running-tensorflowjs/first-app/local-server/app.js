let express = require("express");
let app = express();

const PORT = 2000;

app.get('/', (req, res) => {
	res.sendFile(__dirname + '/View/index.html');
});

app.get('/classify-url', (req, res) => {
	res.sendFile(__dirname + '/View/classify-url.html')
});

app.use(express.static(__dirname + '/public'))

app.listen(PORT, () => {
	console.log(`Serving static on ${PORT}`);
});