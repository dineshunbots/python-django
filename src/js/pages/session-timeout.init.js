/*
Template Name: Upzet -  Admin & Dashboard Template
Author: Themesdesign
Contact: themesdesign.in@gmail.com
File: Session Timeout Js File
*/

$.sessionTimeout({
	keepAliveUrl: '../pages/starter/',
	logoutButton:'Logout',
	logoutUrl: '../account/logout/',
	redirUrl: '../pages/auth-lock-screen',
	warnAfter: 3000,
	redirAfter: 30000,
	keepAlive: true,
	countdownMessage: 'Redirecting in {timer} seconds.'
});