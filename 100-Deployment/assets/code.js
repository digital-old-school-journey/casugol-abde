var age_, gender_, income_, spend_;

$(document).ready(function() {
  // fetch all DOM elements for the input
  age_ = document.getElementById("age");
  gender_ = document.getElementById("gender");
  income_ = document.getElementById("income");
  spend_ = document.getElementById("spend");
});

$(document).on("click", "#btn", function() {
  var age = age_.value;
  var gender = gender_.value;
  var income = income_.value;
  var spend = income_.value;
  if (age == "" || gender == "" || income == "" || spend == "") {
    alert("empty fields not allowed");
  } else {
    //   var requestURL = "https://<username>.pythonanywhere.com/?age="+age+"&gender="+gender+"&income="+income;
    var param = 'age='+age+'&gender='+gender+'&income='+income+'&spend='+spend
    var requestURL = "http://localhost:5000/customer/?"+param;
    var prediction = "";
    // console.log(requestURL); 
    $.getJSON(requestURL, function(data) {
      // console.log(data);
      // console.log(data['cluster']);
      prediction = data['cluster'][0];
      $(".result").html("Prediction is: " + prediction);
    });
  }
});
