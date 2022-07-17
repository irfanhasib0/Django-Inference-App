import React, { useState } from 'react';
import axios from 'axios'
import {Container, Card, Row, Col, Button } from 'react-bootstrap'
import './form.css';
import {AiOutlineMail, AiOutlineLogin, AiOutlineLogout} from 'react-icons/ai';

function Login(props) {
  const [username, setUserName] = useState();
  const [password, setPassword] = useState();

  const getToken = async () => {
  let result = await axios.post('/api/get_token',{'user':username,'password':password})
  console.log("Found token",result.data[0].token)
  if (result.data[0].token) {
    window.localStorage.setItem('token',result.data[0].token)
  }
  
  }
  return(
    <div className="login-wrapper">
      <form>
        <label>
          <p>
          <input placeholder='username' style={{'width': '150px', 'marginRight':'5px'}} type="text" onChange={e => setUserName(e.target.value)}/>
          <input placeholder='password' style={{'width': '150px', 'marginBottom':'0px'}} type="password" onChange={e => setPassword(e.target.value)}/>
          </p>
          <Button variant='outline-info' style={{width : 'auto', height : '25px', padding : '0px', marginRight : '10px' , marginBottom : '5px'}} onClick={getToken} ><AiOutlineLogin/></Button>
        </label>
        </form>
    </div>
  )
}

export default Login;