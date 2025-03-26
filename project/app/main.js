document.addEventListener('DOMContentLoaded', () => {
	const backendUrl = 'http://localhost:8000'

	const uploadBtn = document.getElementById('upload-btn')
	const imageInput = document.getElementById('image-input')
	const resultDiv = document.getElementById('result')

	// To show toast notification
	function showToast(message) {
		const toast = document.createElement('div')
		toast.className = 'toast'
		toast.innerText = message
		document.getElementById('toast-container').appendChild(toast)

		setTimeout(() => {
			toast.style.opacity = '1'
		}, 10)

		setTimeout(() => {
			toast.style.opacity = '0'
			setTimeout(() => {
				toast.remove()
			}, 500)
		}, 3000)
	}

	// Upload Image
	uploadBtn.addEventListener('click', async () => {
		const file = imageInput.files[0]
		if (!file) {
			showToast('Please select an image')
			return
		}

		const formData = new FormData()
		formData.append('image', file)

		try {
			const response = await fetch(`${backendUrl}/upload-image/`, {
				method: 'POST',
				body: formData,
			})
			if (!response.ok) {
				throw new Error(`HTTP error! Status: ${response.status}`)
			}
			const data = await response.json()
			resultDiv.innerHTML = `Predicted Class: ${data.predicted_class}`
			showToast('Image processed successfully')
		} catch (error) {
			console.error('Error uploading image:', error)
			showToast('Failed to process image.')
		}
	})
})
