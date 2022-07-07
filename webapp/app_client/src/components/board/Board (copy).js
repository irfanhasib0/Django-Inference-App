import { ReactSortable } from "react-sortablejs";
    import { useState, useEffect } from "react";
    import axios from "axios";
    
    const Board = () => {
      const [tasks, settasks] = useState([]);
      
      const [ideas, setideas] = useState([]);
      const [todo, settodo] = useState([]);
      const [inprogress, setinprogress] = useState([]);
      const [published, setpublished] = useState([]);
    
      const [newTask, setnewTask] = useState("");
    
      const addTask = async () => {
    
      };
    
      const getTasks = async () => {
    
      };
    
      useEffect(() => {
        getTasks();
      }, []);
    
      return (
        <>
          <div className="container mt-5 mb-5">
            <div
              className="row"
              style={{
                height: "80vh",
              }}
            >
              <div className="col mx-2 px-2 py-3 bg-light border rounded">
                <h6>Idea</h6>
                <div
                  style={{
                    minHeight: "500px",
                  }}
                >
                  
                </div>
                <div>
                  <textarea
                    rows={"1"}
                    cols={30}
                    style={{ float: "left", borderBlockColor: "#007bff" }}
                    value={newTask}
                  ></textarea>
                  <button
                    type="button"
                    style={{ float: "right", marginTop: "2px" }}
                    class="btn btn-primary btn-sm"
                    onClick={addTask}
                  >
                    Add Task
                  </button>
                </div>
              </div>
              <div className="col mx-2 px-2 py-3 bg-light border rounded">
                <h6>Todo</h6>
    
              </div>
              <div className="col mx-2 px-2 py-3 bg-light border rounded">
                <h6>In Progress</h6>
              </div>
              <div className="col mx-2 px-2 py-3 bg-light border rounded">
                <h6>Published</h6>
              </div>
            </div>
          </div>
        </>
      );
    };
    
    export default Board;
