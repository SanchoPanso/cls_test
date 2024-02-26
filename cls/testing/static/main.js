// custom javascript

(function() {
    console.log('Sanity Check!');
  })();
  
  function handleClick(type) {
    input_form = document.getElementById("guid-input");
    console.log(input_form.value);
    input_form.value = "";
    fetch('/tasks', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ type: type }),
    })
    .then(response => response.json())
    .then(data => {
      getStatus(data.task_id)
    })
  }
  
  function getStatus(taskID) {
    fetch(`/tasks/${taskID}`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json'
      },
    })
    .then(response => response.json())
    .then(res => {
      console.log(res)
      const html = `
        <tr>
          <td>${taskID}</td>
          <td>${res.task_status}</td>
          <td>${res.task_result}</td>
        </tr>`;
      // document.getElementById('tasks')
      var tasks = document.getElementById('tasks');
      rows = tasks.getElementsByTagName('tr')
      var found = false;
      for (let i = 0; i < rows.length; i++) {
        if (rows[i].getElementsByTagName('td')[0].innerHTML == taskID) {
          rows[i].innerHTML = html;
          found = true;
          break;
        }
      }
      if (!found) {
        const newRow = document.getElementById('tasks').insertRow(0);
        newRow.innerHTML = html;
      }
      // const newRow = document.getElementById('tasks').insertRow(0);
      // newRow.innerHTML = html;
  
      const taskStatus = res.task_status;
      if (taskStatus === 'SUCCESS' || taskStatus === 'FAILURE') return false;
      setTimeout(function() {
        getStatus(res.task_id);
      }, 1000);
    })
    .catch(err => console.log(err));
  }
  