var express = require('express');
var app = express();
var birds = require('./birds');

var PORT = 2100;
// ...

app.use('/birds', birds);

app.use(express.static('assets'));

app.listen(PORT, () => {
	console.log(`Server is running on ${PORT}`);
});