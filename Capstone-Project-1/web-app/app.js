var express = require('express');
var multer = require('multer');

var upload = multer({dest: 'uploads/'});
var app = express();

const PORT = 2020;

app.use(express.static(__dirname + '/public'));

// MNIST route
app.get('/mnist', (req, res) => {
	res.sendFile(__dirname + '/mnist.html');
});

// web-cam-transfer-learning route
app.get('/web-cam-transfer-learning', (req, res) => {
	res.sendFile(__dirname + '/web-cam-transfer-learning.html');
});

// Index route
app.get('/', (req, res) => {
	res.sendFile(__dirname + '/index.html');
});


// Age and Gender Classification route
app.get('/myApp', (req, res) => {
	res.sendFile(__dirname + '/age-and-gender-classification.html');
});


app.post('/upload', upload.single('picture'), (req, res, next) => {
	console.log('- Got an image from user.');
	console.log('- path of the uploaded image ' + req.file['path']);
	res.end();
});

app.listen(PORT, () => {
	console.log(`Server is running on ${PORT}`);
});