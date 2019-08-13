import io from 'socket.io-client';
const predictContainer = document.getElementById('predictContainer');
const predictButton = document.getElementById('predict-button');
const dataButton = document.getElementById('data-button');

const socket =
    io('http://localhost:8001',
       {reconnectionDelay: 300, reconnectionDelayMax: 300});

//const testSample = [38,31.0,1]; // survived (age, fare, is_female)
//const testSample = [28,7.0,0]; //did not survive

var testSample = [];
var age;
var fare;
var sex;

//const testSample = [40,0.0,0]; //died
dataButton.onclick = () => {
  //dataButton.disabled = true;
  //testSample = document.getElementById('data').value;
  age = parseInt(document.getElementById('age').value);
  testSample.push(age);

  fare = parseInt(document.getElementById('fare').value);
  testSample.push(fare);

  sex = parseInt(document.getElementById('sex').value);
  testSample.push(sex);

  console.log(testSample);
}


predictButton.onclick = () => {
  predictButton.disabled = true;
  socket.emit('predictSample', testSample);
  testSample = [];
};

// functions to handle socket events
socket.on('connect', () => {
    document.getElementById('waiting-msg').style.display = 'none';
    document.getElementById('trainingStatus').innerHTML = 'Training in Progress';
});

socket.on('trainingComplete', () => {
  document.getElementById('trainingStatus').innerHTML = 'Training Complete';
  document.getElementById('predictSample').innerHTML = '[' + testSample.join(', ') + ']';
  predictContainer.style.display = 'block';
});

socket.on('predictResult', (result) => {
  plotPredictResult(result);
});

socket.on('disconnect', () => {
  document.getElementById('trainingStatus').innerHTML = '';
  predictContainer.style.display = 'none';
  document.getElementById('waiting-msg').style.display = 'block';
});

function plotPredictResult(result) {
  predictButton.disabled = false;
  document.getElementById('predictResult').innerHTML = result;
  console.log(result);
}
