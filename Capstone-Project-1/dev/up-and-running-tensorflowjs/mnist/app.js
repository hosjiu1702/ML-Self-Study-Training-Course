var express = require('express');
var app = express();

const PORT = 2020;

app.use(express.static(__dirname + '/public'));

app.get('/', (req, res) => {
	res.sendFile(__dirname + '/index.html');
})

app.listen(PORT, () => {
	console.log(`Server is running on ${PORT}`);
})