var http = require('http');

var testFolder = '/media/';
var fs = require('node:fs');


http.createServer(function (req, res) {
  res.writeHead(200, {'Content-Type': 'text/html'});
  res.write("pqrs ....")
  fs.readdir(testFolder, (err, files) => {
  console.log("abcd ....")
  files.forEach(file => {
    res.write(file);
  });
  res.end('Hello');
});
}).listen(9007);
