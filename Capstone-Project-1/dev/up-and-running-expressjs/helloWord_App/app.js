const express = require('express')
const app = express()
const port = 3000

app.get('/', (req, res) => {
	res.send('Tensorflow is coming soon ...\n GET')
})

app.post('/', (req, res) => {
	res.send('Tensorflow is coming soon ...\n POST')
})

app.put('/', (req, res) => {
	res.send('Tensorflow is coming soon ...\n PUT')
})

app.delete('/', (req, res) => {
	res.send('Tensorflow is coming soon ...\n DELETE')
})

/*-------------------------------------------*/

var cb0 = function (req, res, next) {
  console.log('CB0')
  next()
}

var cb1 = function (req, res, next) {
  console.log('CB1')
  next()
}

var cb2 = function (req, res) {
  res.send('Hello from C!')
}

app.get('/example/c', [cb0, cb1, cb2])


/*------------------------------------------*/
app.route('/book')
  .get(function (req, res) {
    res.send('Get a random book')
  })
  .post(function (req, res) {
    res.send('Add a book')
  })
  .put(function (req, res) {
    res.send('Update the book')
  })


app.listen(port, () => {
	console.log(`Example app listening on port ${port}!`)
})