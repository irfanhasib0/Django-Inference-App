import Board from "./Board";
    
function BoardApp() {
return (
<div
  className="App p-3"
  style={{
    background: "linear-gradient(to right, #0062cc, #007bff)",
  }}
>
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" />
  <Board />
</div>
);
}

export default BoardApp;
