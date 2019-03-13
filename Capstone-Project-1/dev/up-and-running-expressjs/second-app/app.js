const express = require('express');
const app = express();
const path = require('path');
const router = express.Router();

const PORT = 5000;

router.get('/', (req, res) => {
	res.sendFile(path.join(__dirname +  '/static/index.html'));
});

router.get('/about', (req, res) => {
	res.sendFile(path.join(__dirname + '/static/about.html'));
});

router.get('/sitemap', (req, res) => {
	res.sendFile(path.join(__dirname + '/static/sitemap.html'));
});

app.use(express.static(__dirname + '/View'));
app.use(express.static(__dirname + '/Script'));

/* Add the router */
app.use('/', router);

app.listen(PORT,() => {
	console.log(`Server is running on ${PORT}`);
});